/*
** ATOP - System & Process Monitor
**
** The program 'atop' offers the possibility to view the activity of
** the system on system-level as well as process-level.
**
** This source-file contains functions for direct GPU statistics gathering
** using the NVIDIA Management Library (NVML).
** ================================================================
** Author:      Dennis
** E-mail:      gerlof.langeveld@atoptool.nl
** Initial:     March 2025
**
** This program is free software; you can redistribute it and/or modify it
** under the terms of the GNU General Public License as published by the
** Free Software Foundation; either version 2, or (at your option) any
** later version.
**
** This program is distributed in the hope that it will be useful, but
** WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
** See the GNU General Public License for more details.
*/

#include <sys/types.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <nvml.h>

#include "atop.h"
#include "photosyst.h"
#include "photoproc.h"
#include "gpustat.h"

#define COMPUTE      1       // task support bit value
#define ACCOUNT      2       // task support bit value

// Per-GPU process statistics
typedef struct {
    long            pid;
    char            state;
    int             gpubusy;
    int             membusy;
    long long       timems;
    long long       memnow;
    long long       memcum;
    long long       sample;
    struct procstat *next;   // For linked list of per-GPU processes
} procstat_t;

// Per-GPU bookkeeping
typedef struct {
    nvmlDevice_t    gpuhandle;
    char            busid[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    char            devname[NVML_DEVICE_NAME_BUFFER_SIZE];
    int             tasksupport;

    int             gpupercnow;
    int             mempercnow;
    
    long long       memtotalnow;
    long long       memusednow;
    
    long long       gpusamples;
    long long       gpuperccum;
    long long       memperccum;
    long long       memusedcum;
    
    procstat_t      *procstats;     // Active processes using this GPU
    procstat_t      *termprocs;     // Terminated processes using this GPU
} gpuprop_t;

// Global variables
static gpuprop_t    *gpulist = NULL;
static int          numgpus = 0;
static pthread_mutex_t gpulock = PTHREAD_MUTEX_INITIALIZER;
static int          gpu_initialized = 0;
static pthread_t    scanner_thread;
static int          shutdown_requested = 0;

// Function prototypes
static void *gpuscanner(void *arg);
static void update_gpu_stats(gpuprop_t *gpu);
static int compgpupid(const void *, const void *);

/*
** Initialize GPU monitoring by setting up NVML
** and starting a background thread to collect stats
**
** Return value:
**      number of GPUs found
*/
int
gpu_init(void)
{
    nvmlReturn_t ret;

    // Already initialized?
    if (gpu_initialized)
        return numgpus;

    // Try to initialize NVML
    ret = nvmlInit();
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "GPU monitoring not available: %s\n", 
                nvmlErrorString(ret));
        return 0;
    }

    // Get number of GPUs
    ret = nvmlDeviceGetCount(&numgpus);
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "Failed to get GPU count: %s\n", 
                nvmlErrorString(ret));
        nvmlShutdown();
        return 0;
    }

    if (numgpus == 0) {
        nvmlShutdown();
        return 0;
    }

    // Initialize per-GPU bookkeeping
    gpulist = (gpuprop_t*)calloc(numgpus, sizeof(gpuprop_t));
    if (!gpulist) {
        fprintf(stderr, "Memory allocation failed for GPU list\n");
        nvmlShutdown();
        return 0;
    }

    // Initialize each GPU
    for (int i = 0; i < numgpus; i++) {
        ret = nvmlDeviceGetHandleByIndex(i, &gpulist[i].gpuhandle);
        if (ret != NVML_SUCCESS) {
            fprintf(stderr, "Failed to get GPU handle for GPU %d: %s\n", 
                    i, nvmlErrorString(ret));
            continue;
        }

        nvmlDeviceGetPciInfo(gpulist[i].gpuhandle, (nvmlPciInfo_t*)gpulist[i].busid);
        nvmlDeviceGetName(gpulist[i].gpuhandle, gpulist[i].devname, 
                        NVML_DEVICE_NAME_BUFFER_SIZE);
        
        // Replace spaces with underscores in device name
        for (char *p = gpulist[i].devname; *p; p++) {
            if (*p == ' ') *p = '_';
        }
        
        // Check compute support
        nvmlProcessInfo_t procInfo;
        unsigned int procCount = 0;
        if (nvmlDeviceGetComputeRunningProcesses(gpulist[i].gpuhandle, &procCount, &procInfo) == NVML_SUCCESS) {
            gpulist[i].tasksupport |= COMPUTE;
        }
        
        // Check accounting support
        if (nvmlDeviceSetAccountingMode(gpulist[i].gpuhandle, NVML_FEATURE_ENABLED) == NVML_SUCCESS) {
            nvmlDeviceSetPersistenceMode(gpulist[i].gpuhandle, NVML_FEATURE_ENABLED); // NVIDIA advise
            gpulist[i].tasksupport |= ACCOUNT;
        }
    }

    // Start scanner thread to collect stats periodically
    shutdown_requested = 0;
    if (pthread_create(&scanner_thread, NULL, gpuscanner, (void*)1) != 0) {
        fprintf(stderr, "Failed to create GPU scanner thread\n");
        free(gpulist);
        nvmlShutdown();
        return 0;
    }

    gpu_initialized = 1;
    return numgpus;
}

/*
** Clean up GPU monitoring resources
*/
void
gpu_cleanup(void)
{
    if (!gpu_initialized)
        return;

    // Signal scanner thread to terminate and wait for it
    shutdown_requested = 1;
    pthread_join(scanner_thread, NULL);

    // Free GPU list
    if (gpulist) {
        for (int i = 0; i < numgpus; i++) {
            procstat_t *proc = gpulist[i].procstats;
            while (proc) {
                procstat_t *next = proc->next;
                free(proc);
                proc = next;
            }
            
            proc = gpulist[i].termprocs;
            while (proc) {
                procstat_t *next = proc->next;
                free(proc);
                proc = next;
            }
        }
        free(gpulist);
        gpulist = NULL;
    }

    // Shutdown NVML
    nvmlShutdown();
    gpu_initialized = 0;
}

/*
** Thread function to periodically scan all GPUs
*/
static void *
gpuscanner(void *arg)
{
    int interval = (int)(intptr_t)arg;
    
    if (interval < 1)
        interval = 1;

    while (!shutdown_requested) {
        pthread_mutex_lock(&gpulock);
        
        // Update stats for all GPUs
        for (int i = 0; i < numgpus; i++) {
            update_gpu_stats(&gpulist[i]);
        }
        
        pthread_mutex_unlock(&gpulock);
        
        // Sleep for interval
        struct timespec ts;
        ts.tv_sec = interval;
        ts.tv_nsec = 0;
        nanosleep(&ts, NULL);
    }
    
    return NULL;
}

/*
** Update statistics for a GPU
*/
static void
update_gpu_stats(gpuprop_t *gpu)
{
    // Increment sample counter
    gpu->gpusamples++;
    
    // Get utilization rates
    nvmlUtilization_t util;
    if (nvmlDeviceGetUtilizationRates(gpu->gpuhandle, &util) == NVML_SUCCESS) {
        gpu->gpupercnow = util.gpu;
        gpu->mempercnow = util.memory;
        gpu->gpuperccum += util.gpu;
        gpu->memperccum += util.memory;
    } else {
        gpu->gpupercnow = -1;
        gpu->mempercnow = -1;
        gpu->gpuperccum = -1;
        gpu->memperccum = -1;
    }
    
    // Get memory info
    nvmlMemory_t meminfo;
    if (nvmlDeviceGetMemoryInfo(gpu->gpuhandle, &meminfo) == NVML_SUCCESS) {
        gpu->memtotalnow = meminfo.total / 1024;  // Convert to KiB
        gpu->memusednow = meminfo.used / 1024;    // Convert to KiB
        gpu->memusedcum += meminfo.used / 1024;   // Cumulative in KiB
    }
    
    // Get process info if compute is supported
    if (gpu->tasksupport & COMPUTE) {
        nvmlProcessInfo_t *procInfos = NULL;
        unsigned int procCount = 0;
        
        // First call to get the count
        nvmlDeviceGetComputeRunningProcesses(gpu->gpuhandle, &procCount, NULL);
        
        if (procCount > 0) {
            procInfos = (nvmlProcessInfo_t*)malloc(procCount * sizeof(nvmlProcessInfo_t));
            if (!procInfos) {
                return;
            }
            
            // Second call to get the actual data
            nvmlDeviceGetComputeRunningProcesses(gpu->gpuhandle, &procCount, procInfos);
            
            // Build list of currently active processes
            procstat_t *activeProcs = NULL;
            int numActiveProcs = 0;
            procstat_t *proc = gpu->procstats;
            
            while (proc) {
                activeProcs = (procstat_t*)realloc(activeProcs, (numActiveProcs + 1) * sizeof(procstat_t));
                if (!activeProcs) {
                    free(procInfos);
                    return;
                }
                activeProcs[numActiveProcs++].pid = proc->pid;
                proc = proc->next;
            }
            
            // Process each running process
            for (unsigned int i = 0; i < procCount; i++) {
                int found = 0;
                proc = gpu->procstats;
                procstat_t *prev = NULL;
                
                // Check if process is already in our list
                while (proc && !found) {
                    if (proc->pid == procInfos[i].pid) {
                        found = 1;
                        
                        // Update stats
                        if (procInfos[i].usedGpuMemory) {
                            proc->memnow = procInfos[i].usedGpuMemory / 1024;  // KiB
                            proc->memcum += procInfos[i].usedGpuMemory / 1024;
                            proc->sample++;
                        }
                        
                        if (gpu->tasksupport & ACCOUNT) {
                            nvmlAccountingStats_t stats;
                            if (nvmlDeviceGetAccountingStats(gpu->gpuhandle, procInfos[i].pid, &stats) == NVML_SUCCESS) {
                                proc->gpubusy = stats.gpuUtilization;
                                proc->membusy = stats.memoryUtilization;
                                proc->timems = stats.time;
                            }
                        }
                    }
                    
                    prev = proc;
                    proc = proc->next;
                }
                
                // New process, create entry
                if (!found) {
                    procstat_t *newProc = (procstat_t*)calloc(1, sizeof(procstat_t));
                    if (!newProc) {
                        continue;
                    }
                    
                    newProc->pid = procInfos[i].pid;
                    newProc->state = 'A';  // Active
                    newProc->gpubusy = -1;
                    newProc->membusy = -1;
                    newProc->timems = -1;
                    
                    if (procInfos[i].usedGpuMemory) {
                        newProc->memnow = procInfos[i].usedGpuMemory / 1024;  // KiB
                        newProc->memcum = procInfos[i].usedGpuMemory / 1024;
                        newProc->sample = 1;
                    }
                    
                    if (gpu->tasksupport & ACCOUNT) {
                        nvmlAccountingStats_t stats;
                        if (nvmlDeviceGetAccountingStats(gpu->gpuhandle, procInfos[i].pid, &stats) == NVML_SUCCESS) {
                            newProc->gpubusy = stats.gpuUtilization;
                            newProc->membusy = stats.memoryUtilization;
                            newProc->timems = stats.time;
                        }
                    }
                    
                    // Add to list
                    newProc->next = gpu->procstats;
                    gpu->procstats = newProc;
                }
                
                // Remove from active processes list
                for (int j = 0; j < numActiveProcs; j++) {
                    if (activeProcs[j].pid == procInfos[i].pid) {
                        if (j < numActiveProcs - 1) {
                            memmove(&activeProcs[j], &activeProcs[j+1], 
                                    (numActiveProcs - j - 1) * sizeof(procstat_t));
                        }
                        numActiveProcs--;
                        break;
                    }
                }
            }
            
            // Handle terminated processes
            for (int i = 0; i < numActiveProcs; i++) {
                proc = gpu->procstats;
                procstat_t *prev = NULL;
                
                // Find the process in our list
                while (proc) {
                    if (proc->pid == activeProcs[i].pid) {
                        // Process terminated, remove from active list
                        if (prev) {
                            prev->next = proc->next;
                        } else {
                            gpu->procstats = proc->next;
                        }
                        
                        // Mark as terminated and add to termprocs list
                        proc->state = 'E';  // Exit
                        proc->next = gpu->termprocs;
                        gpu->termprocs = proc;
                        break;
                    }
                    
                    prev = proc;
                    proc = proc->next;
                }
            }
            
            if (activeProcs) {
                free(activeProcs);
            }
            
            free(procInfos);
        }
    }
}

/*
** Get system-level GPU statistics
**
** Parameters:
**      maxgpu    maximum number of GPUs to handle
**      ggs       array to fill with GPU statistics
**
** Return value:
**      number of GPUs handled
*/
int
gpu_getsysstat(int maxgpu, struct pergpu *ggs)
{
    if (!gpu_initialized || !gpulist || numgpus == 0)
        return 0;

    int count = numgpus < maxgpu ? numgpus : maxgpu;
    
    pthread_mutex_lock(&gpulock);
    
    for (int i = 0; i < count; i++) {
        gpuprop_t *gpu = &gpulist[i];
        struct pergpu *gg = &ggs[i];
        
        // Copy system-level stats
        gg->gpupercnow  = gpu->gpupercnow;
        gg->mempercnow  = gpu->mempercnow;
        gg->memtotnow   = gpu->memtotalnow;
        gg->memusenow   = gpu->memusednow;
        gg->samples     = gpu->gpusamples;
        gg->gpuperccum  = gpu->gpuperccum;
        gg->memperccum  = gpu->memperccum;
        gg->memusecum   = gpu->memusedcum;
        gg->gpunr       = i;
        
        // Copy device info
        strncpy(gg->type, gpu->devname, MAXGPUTYPE);
        gg->type[MAXGPUTYPE] = '\0';
        
        strncpy(gg->busid, gpu->busid, MAXGPUBUS);
        gg->busid[MAXGPUBUS] = '\0';
        
        // Process count
        gg->nrprocs = 0;
        procstat_t *proc = gpu->procstats;
        while (proc) {
            gg->nrprocs++;
            proc = proc->next;
        }
        
        // Task support
        gg->taskstats = gpu->tasksupport;
    }
    
    pthread_mutex_unlock(&gpulock);
    
    return count;
}

/*
** Get process-level GPU statistics
**
** Parameters:
**      gps       pointer to array of gpupidstat structures to fill
**
** Return value:
**      number of process entries
*/
int
gpu_getprocstat(struct gpupidstat **gps)
{
    if (!gpu_initialized || !gpulist || numgpus == 0)
        return 0;

    int pidcount = 0;
    
    // First, count the total number of processes
    pthread_mutex_lock(&gpulock);
    
    for (int i = 0; i < numgpus; i++) {
        procstat_t *proc = gpulist[i].procstats;
        while (proc) {
            pidcount++;
            proc = proc->next;
        }
        
        proc = gpulist[i].termprocs;
        while (proc) {
            pidcount++;
            proc = proc->next;
        }
    }
    
    if (pidcount == 0) {
        pthread_mutex_unlock(&gpulock);
        *gps = NULL;
        return 0;
    }
    
    // Allocate array for process stats
    *gps = malloc(pidcount * sizeof(struct gpupidstat));
    if (!*gps) {
        pthread_mutex_unlock(&gpulock);
        return 0;
    }
    
    // Fill the array with process stats
    int idx = 0;
    
    for (int i = 0; i < numgpus; i++) {
        // Active processes
        procstat_t *proc = gpulist[i].procstats;
        while (proc) {
            struct gpupidstat *gp = &(*gps)[idx++];
            
            gp->pid = proc->pid;
            gp->gpu.state = proc->state;
            gp->gpu.gpubusy = proc->gpubusy;
            gp->gpu.membusy = proc->membusy;
            gp->gpu.timems = proc->timems;
            gp->gpu.memnow = proc->memnow;
            gp->gpu.memcum = proc->memcum;
            gp->gpu.sample = proc->sample;
            
            gp->gpu.nrgpus = 1;
            gp->gpu.gpulist = 1 << i;
            
            proc = proc->next;
        }
        
        // Terminated processes
        proc = gpulist[i].termprocs;
        procstat_t *prev = NULL;
        
        while (proc) {
            struct gpupidstat *gp = &(*gps)[idx++];
            
            gp->pid = proc->pid;
            gp->gpu.state = proc->state;
            gp->gpu.gpubusy = proc->gpubusy;
            gp->gpu.membusy = proc->membusy;
            gp->gpu.timems = proc->timems;
            gp->gpu.memnow = proc->memnow;
            gp->gpu.memcum = proc->memcum;
            gp->gpu.sample = proc->sample;
            
            gp->gpu.nrgpus = 1;
            gp->gpu.gpulist = 1 << i;
            
            // Remove from terminated list
            procstat_t *next = proc->next;
            
            if (prev) {
                prev->next = next;
            } else {
                gpulist[i].termprocs = next;
            }
            
            free(proc);
            proc = next;
        }
    }
    
    pthread_mutex_unlock(&gpulock);
    
    // Sort by PID for consistent ordering
    if (pidcount > 1) {
        qsort(*gps, pidcount, sizeof(struct gpupidstat), compgpupid);
    }
    
    // Merge entries for the same PID
    for (int i = 1; i < pidcount; i++) {
        if ((*gps)[i-1].pid == (*gps)[i].pid) {
            struct gpupidstat *p = &(*gps)[i-1];
            struct gpupidstat *q = &(*gps)[i];
            
            p->gpu.nrgpus  += q->gpu.nrgpus;
            p->gpu.gpulist |= q->gpu.gpulist;
            
            if (p->gpu.gpubusy != -1)
                p->gpu.gpubusy += q->gpu.gpubusy;
            
            if (p->gpu.membusy != -1)
                p->gpu.membusy += q->gpu.membusy;
            
            if (p->gpu.timems != -1)
                p->gpu.timems += q->gpu.timems;
            
            p->gpu.memnow += q->gpu.memnow;
            p->gpu.memcum += q->gpu.memcum;
            p->gpu.sample += q->gpu.sample;
            
            if (pidcount-i-1 > 0)
                memmove(&(*gps)[i], &(*gps)[i+1], (pidcount-i-1) * sizeof(struct gpupidstat));
            
            pidcount--;
            i--;
        }
    }
    
    return pidcount;
}

/*
** Merge the GPU per-process counters with the other
** per-process counters (identical to the original gpumergeproc in gpucom.c)
*/
void
gpu_mergeproc(struct tstat      *curtpres, int ntaskpres,
              struct tstat      *curpexit, int nprocexit,
              struct gpupidstat *gpuproc,  int nrgpuproc)
{
    struct gpupidstat  **gpp;
    int                 t, g, gpuleft = nrgpuproc;

    /*
     ** make pointer list for elements in gpuproc
     */
    gpp = malloc(nrgpuproc * sizeof(struct gpupidstat *));

    if (!gpp)
        return;

    for (g=0; g < nrgpuproc; g++)
        gpp[g] = gpuproc + g;

    /*
       ** sort the list with pointers in order of pid
     */
    if (nrgpuproc > 1)
        qsort(gpp, nrgpuproc, sizeof(struct gpupidstat *), compgpupid);

    /*
     ** merge gpu stats with sorted task list of active processes
     */
    for (t=g=0; t < ntaskpres && g < nrgpuproc; t++)
    {
        if (curtpres[t].gen.isproc)
        {
            if (curtpres[t].gen.pid == gpp[g]->pid)
            {
                curtpres[t].gpu = gpp[g]->gpu;
                gpp[g++] = NULL;

                if (--gpuleft == 0 || g >= nrgpuproc)
                    break;
            }

            // anyhow resync
            while (g < nrgpuproc && curtpres[t].gen.pid > gpp[g]->pid)
            {
                if (++g >= nrgpuproc)
                    break;
            }
        }
    }

    if (gpuleft == 0)
    {
        free(gpp);
        return;
    }

    /*
     ** compact list with pointers to remaining pids
     */
    for (g=t=0; g < nrgpuproc; g++)
    {
        if (gpp[g] == NULL)
        {
            for (; t < nrgpuproc; t++)
            {
                if (gpp[t])
                {
                    gpp[g] = gpp[t];
                    gpp[t] = NULL;
                    break;
                }
            }
        }
    }

    /*
     ** merge remaining gpu stats with task list of exited processes
     */
    for (t=0; t < nprocexit && gpuleft; t++)
    {
        if (curpexit[t].gen.isproc)
        {
            for (g=0; g < gpuleft; g++)
            {
                if (gpp[g] == NULL)
                    continue;

                if (curpexit[t].gen.pid == gpp[g]->pid)
                {
                    curpexit[t].gpu = gpp[g]->gpu;
                    gpp[g] = NULL;
                    gpuleft--;
                }
            }
        }
    }

    free(gpp);
}

static int
compgpupid(const void *a, const void *b)
{
    return ((struct gpupidstat *)a)->pid - ((struct gpupidstat *)b)->pid;
}
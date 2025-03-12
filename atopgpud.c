/*
** ATOP - System & Process Monitor 
**
** Daemon that gathers statistical information from all
** Nvidia GPUs in the current system. Every second, it gathers
** the statistics of every GPU and maintains cumulative counters,
** globally and per process.
**
** Clients can connect to this daemon on TCP port 59123.
** Clients can send requests of two bytes, consisting of one byte
** request code followed by one byte integer version number.
** The request code can be 'T' to obtain the GPU types or 'S' to
** obtain all statistical counters.
**
** The response of the daemon starts with a 4-byte integer. The
** first byte is the version of the response format and the
** subsequent three bytes indicate the length (big endian) of the
** response string that follows.
** ================================================================
** Author:      Original by Gerlof Langeveld, native C version by Dennis
** E-mail:      gerlof.langeveld@atoptool.nl
** Date:        July 2018 (initial python version), March 2025 (native C version)
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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <syslog.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <nvml.h>
#include "atopgpud.h"

#define GPUDPORT     59123
#define MAXRESPONSE  8192
#define APIVERSION   1

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
} gpuprop_t;

// Client bookkeeping for terminated processes
typedef struct {
    int             socket;         // Client socket descriptor
    procstat_t      *termprocs;     // List of terminated processes for this client
    struct clisock  *next;
} clisock_t;

// Global variables
static gpuprop_t    *gpulist = NULL;
static int          numgpus = 0;
static pthread_mutex_t gpulock = PTHREAD_MUTEX_INITIALIZER;
static clisock_t    *clients = NULL;

// Function prototypes
static void gpuscanner(int interval);
static void serveclient(int sock, struct sockaddr *peeraddr);
static char* format_gpu_types_v1(void);
static char* format_gpu_stats_v1(int clisock);
static void daemonize(void);
static void cleanup(void);
static void update_gpu_stats(gpuprop_t *gpu);
static void handle_signal(int sig);

/*
** Main function
*/
int 
main(int argc, char *argv[])
{
    int sock, optval = 1;
    struct sockaddr_in6 servaddr;
    struct sockaddr cliaddr;
    socklen_t addrlen = sizeof(cliaddr);
    int loglevel = LOG_INFO;
    
    // Initialize logging
    if (argc > 1 && strcmp(argv[1], "-v") == 0)
        loglevel = LOG_DEBUG;
    
    openlog("atopgpud", LOG_PID, LOG_DAEMON);
    setlogmask(LOG_UPTO(loglevel));
    
    // Initialize NVML
    nvmlReturn_t ret = nvmlInit();
    if (ret != NVML_SUCCESS) {
        syslog(LOG_ERR, "Failed to initialize NVML: %s", nvmlErrorString(ret));
        return 1;
    }
    
    // Create IPv6 socket
    if ((sock = socket(AF_INET6, SOCK_STREAM, 0)) < 0) {
        // Try IPv4 if IPv6 fails
        if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
            syslog(LOG_ERR, "Socket creation failed: %s", strerror(errno));
            nvmlShutdown();
            return 1;
        }
    }
    
    // Set socket options
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
    
    // Bind to local port
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin6_family = AF_INET6;
    servaddr.sin6_addr = in6addr_any;
    servaddr.sin6_port = htons(GPUDPORT);
    
    if (bind(sock, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
        syslog(LOG_ERR, "Socket binding to port %d failed: %s", 
               GPUDPORT, strerror(errno));
        close(sock);
        nvmlShutdown();
        return 1;
    }
    
    // Listen for connections
    if (listen(sock, 32) < 0) {
        syslog(LOG_ERR, "Listen failed: %s", strerror(errno));
        close(sock);
        nvmlShutdown();
        return 1;
    }
    
    // Daemonize
    daemonize();
    
    // Re-initialize NVML for the child process
    ret = nvmlInit();
    if (ret != NVML_SUCCESS) {
        syslog(LOG_ERR, "Child failed to initialize NVML: %s", nvmlErrorString(ret));
        return 1;
    }
    
    // Get number of GPUs
    ret = nvmlDeviceGetCount(&numgpus);
    if (ret != NVML_SUCCESS) {
        syslog(LOG_ERR, "Failed to get device count: %s", nvmlErrorString(ret));
        nvmlShutdown();
        return 1;
    }
    
    syslog(LOG_INFO, "Number of GPUs: %d", numgpus);
    
    if (numgpus == 0) {
        syslog(LOG_INFO, "Terminated (no GPUs available)");
        nvmlShutdown();
        return 0;
    }
    
    // Initialize per-GPU bookkeeping
    gpulist = (gpuprop_t*)calloc(numgpus, sizeof(gpuprop_t));
    if (!gpulist) {
        syslog(LOG_ERR, "Memory allocation failed for GPU list");
        nvmlShutdown();
        return 1;
    }
    
    for (int i = 0; i < numgpus; i++) {
        nvmlDeviceGetHandleByIndex(i, &gpulist[i].gpuhandle);
        nvmlDeviceGetPciInfo(gpulist[i].gpuhandle, gpulist[i].busid);
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
    
    // Set up signal handlers
    signal(SIGTERM, handle_signal);
    signal(SIGINT, handle_signal);
    
    // Start scanner thread
    pthread_t scanner_thread;
    if (pthread_create(&scanner_thread, NULL, (void *(*)(void *))gpuscanner, (void*)1) != 0) {
        syslog(LOG_ERR, "Failed to create scanner thread");
        cleanup();
        return 1;
    }
    pthread_detach(scanner_thread);
    
    syslog(LOG_INFO, "Initialization succeeded");
    
    // Main loop - accept clients
    while (1) {
        int newsock = accept(sock, &cliaddr, &addrlen);
        if (newsock < 0) {
            syslog(LOG_ERR, "Accept failed: %s", strerror(errno));
            continue;
        }
        
        // Create a new client thread
        pthread_t client_thread;
        if (pthread_create(&client_thread, NULL, 
                         (void *(*)(void *))serveclient, 
                         (void*)(intptr_t)newsock) != 0) {
            syslog(LOG_ERR, "Failed to create client thread");
            close(newsock);
            continue;
        }
        pthread_detach(client_thread);
    }
    
    // Should never reach here
    cleanup();
    return 0;
}

/*
** Clean up resources
*/
static void
cleanup(void)
{
    // Free GPU list
    if (gpulist) {
        for (int i = 0; i < numgpus; i++) {
            procstat_t *proc = gpulist[i].procstats;
            while (proc) {
                procstat_t *next = proc->next;
                free(proc);
                proc = next;
            }
        }
        free(gpulist);
    }
    
    // Clean up client list
    clisock_t *client = clients;
    while (client) {
        clisock_t *next = client->next;
        procstat_t *proc = client->termprocs;
        while (proc) {
            procstat_t *next_proc = proc->next;
            free(proc);
            proc = next_proc;
        }
        if (client->socket > 0)
            close(client->socket);
        free(client);
        client = next;
    }
    
    // Shutdown NVML
    nvmlShutdown();
}

/*
** Signal handler
*/
static void
handle_signal(int sig)
{
    syslog(LOG_INFO, "Received signal %d, shutting down", sig);
    cleanup();
    exit(0);
}

/*
** Daemonize the process
*/
static void
daemonize(void)
{
    pid_t pid = fork();
    
    if (pid < 0) {
        syslog(LOG_ERR, "Failed to fork child");
        exit(1);
    }
    
    if (pid > 0) {
        // Parent process, exit
        exit(0);
    }
    
    // Child process continues
    umask(0);
    
    if (setsid() < 0) {
        syslog(LOG_ERR, "Failed to create new session");
        exit(1);
    }
    
    // Close standard file descriptors
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);
}

/*
** Thread function to periodically scan all GPUs
*/
static void
gpuscanner(int interval)
{
    while (1) {
        pthread_mutex_lock(&gpulock);
        
        // Update stats for all GPUs
        for (int i = 0; i < numgpus; i++) {
            update_gpu_stats(&gpulist[i]);
        }
        
        pthread_mutex_unlock(&gpulock);
        
        sleep(interval);
    }
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
                syslog(LOG_ERR, "Memory allocation failed for process info");
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
                        syslog(LOG_ERR, "Memory allocation failed for new process entry");
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
                        // Process terminated, move to client lists
                        if (prev) {
                            prev->next = proc->next;
                        } else {
                            gpu->procstats = proc->next;
                        }
                        
                        // Add terminated process to all client termproc lists
                        clisock_t *client = clients;
                        while (client) {
                            procstat_t *termProc = (procstat_t*)malloc(sizeof(procstat_t));
                            if (!termProc) {
                                syslog(LOG_ERR, "Memory allocation failed for terminated process");
                                break;
                            }
                            
                            // Copy stats
                            *termProc = *proc;
                            termProc->state = 'E';  // Exit
                            termProc->next = client->termprocs;
                            client->termprocs = termProc;
                            
                            client = client->next;
                        }
                        
                        free(proc);
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
** Thread function to serve a client
*/
static void
serveclient(int sock, struct sockaddr *peeraddr)
{
    char req[2];
    char *response;
    uint32_t prelude;
    int version, length;
    
    // Create client bookkeeping
    clisock_t *client = (clisock_t*)calloc(1, sizeof(clisock_t));
    if (!client) {
        syslog(LOG_ERR, "Memory allocation failed for client");
        close(sock);
        return;
    }
    
    client->socket = sock;
    
    // Add to client list
    pthread_mutex_lock(&gpulock);
    client->next = clients;
    clients = client;
    pthread_mutex_unlock(&gpulock);
    
    // Set receive timeout (2 seconds)
    struct timeval timeout = {2, 0};
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    
    while (1) {
        // Receive request
        int n = recv(sock, req, sizeof(req), 0);
        if (n <= 0) {
            if (n < 0)
                syslog(LOG_ERR, "Receive error: %s", strerror(errno));
            break;
        }
        
        // Check request length
        if (n != 2) {
            syslog(LOG_ERR, "Wrong request length: %d", n);
            break;
        }
        
        // Process request
        version = (unsigned char)req[1];
        
        if (req[0] == 'T') {
            // Request for GPU types
            if (version == 0 || version > 1)
                version = 1;
            
            response = format_gpu_types_v1();
        }
        else if (req[0] == 'S') {
            // Request for GPU statistics
            if (version == 0 || version > 1)
                version = 1;
            
            response = format_gpu_stats_v1(sock);
        }
        else {
            syslog(LOG_ERR, "Wrong request from client: %c", req[0]);
            break;
        }
        
        // Send response if we have one
        if (response) {
            length = strlen(response);
            prelude = htonl((version << 24) | length);
            
            // Send prelude (version + length)
            if (send(sock, &prelude, sizeof(prelude), 0) < 0) {
                syslog(LOG_ERR, "Send prelude error: %s", strerror(errno));
                free(response);
                break;
            }
            
            // Send response data
            if (send(sock, response, length, 0) < 0) {
                syslog(LOG_ERR, "Send data error: %s", strerror(errno));
                free(response);
                break;
            }
            
            free(response);
        }
    }
    
    // Clean up
    close(sock);
    
    // Remove from client list
    pthread_mutex_lock(&gpulock);
    if (clients == client) {
        clients = client->next;
    } else {
        clisock_t *prev = clients;
        while (prev && prev->next != client) {
            prev = prev->next;
        }
        if (prev) {
            prev->next = client->next;
        }
    }
    pthread_mutex_unlock(&gpulock);
    
    // Free terminated process list
    procstat_t *proc = client->termprocs;
    while (proc) {
        procstat_t *next = proc->next;
        free(proc);
        proc = next;
    }
    
    free(client);
}

/*
** Format GPU types as a string (v1 format)
*/
static char*
format_gpu_types_v1(void)
{
    char *buffer = (char*)malloc(MAXRESPONSE);
    if (!buffer) {
        syslog(LOG_ERR, "Memory allocation failed for types response");
        return NULL;
    }
    
    // Format: numgpus@busid devname tasksupport@busid devname tasksupport@...
    int offset = snprintf(buffer, MAXRESPONSE, "%d", numgpus);
    
    pthread_mutex_lock(&gpulock);
    
    for (int i = 0; i < numgpus; i++) {
        offset += snprintf(buffer + offset, MAXRESPONSE - offset,
                         "@%s %s %d",
                         gpulist[i].busid,
                         gpulist[i].devname,
                         gpulist[i].tasksupport);
    }
    
    pthread_mutex_unlock(&gpulock);
    
    return buffer;
}

/*
** Format GPU statistics as a string (v1 format)
*/
static char*
format_gpu_stats_v1(int clisock)
{
    char *buffer = (char*)malloc(MAXRESPONSE);
    if (!buffer) {
        syslog(LOG_ERR, "Memory allocation failed for stats response");
        return NULL;
    }
    
    buffer[0] = '\0';
    int offset = 0;
    
    pthread_mutex_lock(&gpulock);
    
    // Find the client structure
    clisock_t *client = clients;
    while (client && client->socket != clisock) {
        client = client->next;
    }
    
    // Format stats for each GPU
    for (int i = 0; i < numgpus; i++) {
        gpuprop_t *gpu = &gpulist[i];
        
        // Format: @gpupercnow mempercnow memtotalnow memusednow gpusamples gpuperccum memperccum memusedcum
        offset += snprintf(buffer + offset, MAXRESPONSE - offset,
                         "@%d %d %lld %lld %lld %lld %lld %lld",
                         gpu->gpupercnow, gpu->mempercnow,
                         gpu->memtotalnow, gpu->memusednow, gpu->gpusamples,
                         gpu->gpuperccum, gpu->memperccum, gpu->memusedcum);
        
        // Add active processes for this GPU
        procstat_t *proc = gpu->procstats;
        while (proc) {
            // Format: #A pid gpubusy membusy timems memnow memcum sample
            offset += snprintf(buffer + offset, MAXRESPONSE - offset,
                             "#A %ld %d %d %lld %lld %lld %lld",
                             proc->pid, proc->gpubusy, proc->membusy, proc->timems,
                             proc->memnow, proc->memcum, proc->sample);
            
            proc = proc->next;
        }
        
        // Add terminated processes for this GPU
        if (client) {
            proc = client->termprocs;
            while (proc) {
                // Format: #E pid gpubusy membusy timems memnow memcum sample
                offset += snprintf(buffer + offset, MAXRESPONSE - offset,
                                 "#E %ld %d %d %lld %lld %lld %lld",
                                 proc->pid, proc->gpubusy, proc->membusy, proc->timems,
                                 proc->memnow, proc->memcum, proc->sample);
                
                procstat_t *next = proc->next;
                free(proc);
                proc = next;
            }
            
            // Clear terminated processes for this client
            client->termprocs = NULL;
        }
    }
    
    pthread_mutex_unlock(&gpulock);
    
    return buffer;
}
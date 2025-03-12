/*
** ATOP - System & Process Monitor
**
** The program 'atop' offers the possibility to view the activity of 
** the system on system-level as well as process-level.
** ==========================================================================
** Author:      Dennis
** E-mail:      gerlof.langeveld@atoptool.nl
** Date:        March 2025
** --------------------------------------------------------------------------
** Copyright (C) 2000-2010 Gerlof Langeveld
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
**
** You should have received a copy of the GNU General Public License
** along with this program; if not, write to the Free Software
** Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
** --------------------------------------------------------------------------
*/

#ifndef	__GPUSTAT__
#define	__GPUSTAT__

#include "photoproc.h" // For struct gpu
#include "photosyst.h" // For struct pergpu

struct gpupidstat {
	long		pid;
	struct gpu	gpu;
};

int	gpu_init(void);
void	gpu_cleanup(void);
int	gpu_getsysstat(int maxgpu, struct pergpu *ggs);
int	gpu_getprocstat(struct gpupidstat **gps);
void	gpu_mergeproc(struct tstat      *curtpres, int ntaskpres,
		      struct tstat      *curpexit, int nprocexit,
	              struct gpupidstat *gpuproc,  int nrgpuproc);

#endif /* __GPUSTAT__ */
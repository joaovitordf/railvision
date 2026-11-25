import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { RuntimeStatusService } from '../../../core/services/runtime-status.service';
import { environment } from '../../../../environments/environment';
import {
  CurateFrameRequest,
  CurationExport,
  ImageRunResult,
  LabRunResult,
  ModelCheckpoint,
  RunComparison,
  RunListItem,
  RuntimeStatus,
  VideoRunStatus,
} from '../models/lab.models';

@Injectable({ providedIn: 'root' })
export class LabApiService {
  private readonly http = inject(HttpClient);
  private readonly runtimeStatus = inject(RuntimeStatusService);
  private readonly baseUrl = `${environment.apiBaseUrl}/vision-lab`;

  getStatus(modelId?: string | null): Observable<RuntimeStatus> {
    if (!modelId) {
      return this.runtimeStatus.getStatus();
    }
    const params = new HttpParams().set('modelId', modelId);
    return this.http.get<RuntimeStatus>(`${this.baseUrl}/status`, { params });
  }

  listCheckpoints(): Observable<ModelCheckpoint[]> {
    return this.runtimeStatus.listCheckpoints();
  }

  analyzeImage(
    file: File,
    conf: number,
    iou: number,
    runtime?: string | null,
    allowCpuFallback?: boolean,
    modelId?: string | null
  ): Observable<ImageRunResult> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    let params = new HttpParams().set('conf', String(conf)).set('iou', String(iou));
    if (runtime) {
      params = params.set('runtime', runtime);
    }
    if (allowCpuFallback) {
      params = params.set('allowCpuFallback', 'true');
    }
    if (modelId) {
      params = params.set('modelId', modelId);
    }
    return this.http.post<ImageRunResult>(`${this.baseUrl}/analyze-image`, formData, { params });
  }

  analyzeVideo(
    file: File,
    conf: number,
    iou: number,
    sampleFps: number | null,
    runtime?: string | null,
    outputResolution?: string | null,
    enableTracking?: boolean,
    allowCpuFallback?: boolean,
    modelId?: string | null
  ): Observable<VideoRunStatus> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    let params = new HttpParams().set('conf', String(conf)).set('iou', String(iou));
    if (sampleFps != null && sampleFps > 0) {
      params = params.set('sampleFps', String(sampleFps));
    }
    if (runtime) {
      params = params.set('runtime', runtime);
    }
    if (allowCpuFallback) {
      params = params.set('allowCpuFallback', 'true');
    }
    if (outputResolution) {
      params = params.set('outputResolution', outputResolution);
    }
    if (enableTracking) {
      params = params.set('enableTracking', 'true');
    }
    if (modelId) {
      params = params.set('modelId', modelId);
    }
    return this.http.post<VideoRunStatus>(`${this.baseUrl}/analyze-video`, formData, { params });
  }

  listRuns(): Observable<RunListItem[]> {
    return this.http.get<RunListItem[]>(`${this.baseUrl}/runs`);
  }

  getRun(runId: string): Observable<LabRunResult> {
    return this.http.get<LabRunResult>(`${this.baseUrl}/runs/${encodeURIComponent(runId)}`);
  }

  compareRuns(runA: string, runB: string): Observable<RunComparison> {
    const params = new HttpParams().set('runA', runA).set('runB', runB);
    return this.http.get<RunComparison>(`${this.baseUrl}/compare`, { params });
  }

  curateFrame(runId: string, payload: CurateFrameRequest): Observable<CurationExport> {
    return this.http.post<CurationExport>(
      `${this.baseUrl}/runs/${encodeURIComponent(runId)}/curate-frame`,
      payload
    );
  }
}

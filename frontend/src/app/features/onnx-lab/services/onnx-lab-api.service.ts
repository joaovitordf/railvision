import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../../../environments/environment';
import {
  ImageRunResult,
  LabRunResult,
  RunListItem,
  RuntimeStatus,
  VideoRunStatus,
} from '../models/onnx-lab.models';

@Injectable({ providedIn: 'root' })
export class OnnxLabApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiBaseUrl}/onnx-spike`;

  getStatus(): Observable<RuntimeStatus> {
    return this.http.get<RuntimeStatus>(`${this.baseUrl}/status`);
  }

  analyzeImage(
    file: File,
    conf: number,
    iou: number,
    runtime?: string | null
  ): Observable<ImageRunResult> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    let params = new HttpParams().set('conf', String(conf)).set('iou', String(iou));
    if (runtime) {
      params = params.set('runtime', runtime);
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
    enableTracking?: boolean
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
    if (outputResolution) {
      params = params.set('outputResolution', outputResolution);
    }
    if (enableTracking) {
      params = params.set('enableTracking', 'true');
    }
    return this.http.post<VideoRunStatus>(`${this.baseUrl}/analyze-video`, formData, { params });
  }

  listRuns(): Observable<RunListItem[]> {
    return this.http.get<RunListItem[]>(`${this.baseUrl}/runs`);
  }

  getRun(runId: string): Observable<LabRunResult> {
    return this.http.get<LabRunResult>(`${this.baseUrl}/runs/${encodeURIComponent(runId)}`);
  }
}

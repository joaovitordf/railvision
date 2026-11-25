import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../../../environments/environment';
import {
  AnnotationItemDetail,
  AnnotationItemSummary,
  AnnotationProjectDetail,
  AnnotationProjectSummary,
  AnnotationVideo,
  CreateAnnotationProjectRequest,
  ImportCurationResponse,
  PreannotateBatchResponse,
  PreviewFramesResponse,
  UpdateItemAnnotationsRequest,
} from '../models/annotations.models';

@Injectable({ providedIn: 'root' })
export class AnnotationsApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiBaseUrl}/annotations`;

  listProjects(): Observable<AnnotationProjectSummary[]> {
    return this.http.get<AnnotationProjectSummary[]>(`${this.baseUrl}/projects`);
  }

  createProject(payload: CreateAnnotationProjectRequest): Observable<AnnotationProjectSummary> {
    return this.http.post<AnnotationProjectSummary>(`${this.baseUrl}/projects`, payload);
  }

  getProject(projectId: string, status?: string | null, search?: string | null): Observable<AnnotationProjectDetail> {
    let params = new HttpParams();
    if (status && status !== 'ALL') {
      params = params.set('status', status);
    }
    if (search && search.trim().length > 0) {
      params = params.set('search', search.trim());
    }
    return this.http.get<AnnotationProjectDetail>(`${this.baseUrl}/projects/${encodeURIComponent(projectId)}`, {
      params,
    });
  }

  deleteProject(projectId: string): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/projects/${encodeURIComponent(projectId)}`);
  }

  uploadImages(
    projectId: string,
    files: File[],
    sourceOrigin?: string,
    campaignId?: string
  ): Observable<AnnotationItemSummary[]> {
    const form = new FormData();
    for (const f of files) {
      form.append('files', f, f.name);
    }
    if (sourceOrigin?.trim()) {
      form.append('sourceOrigin', sourceOrigin.trim());
    }
    if (campaignId?.trim()) {
      form.append('campaignId', campaignId.trim());
    }
    return this.http.post<AnnotationItemSummary[]>(
      `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/images`,
      form
    );
  }

  uploadVideo(
    projectId: string,
    file: File,
    sourceOrigin?: string,
    campaignId?: string
  ): Observable<AnnotationVideo> {
    const form = new FormData();
    form.append('file', file, file.name);
    if (sourceOrigin?.trim()) {
      form.append('sourceOrigin', sourceOrigin.trim());
    }
    if (campaignId?.trim()) {
      form.append('campaignId', campaignId.trim());
    }
    return this.http.post<AnnotationVideo>(
      `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/videos`,
      form
    );
  }

  previewVideoFrames(
    projectId: string,
    videoId: string,
    sampleFps: number,
    maxFrames: number,
    dedupThreshold: number
  ): Observable<PreviewFramesResponse> {
    return this.http.post<PreviewFramesResponse>(
      `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/videos/${encodeURIComponent(videoId)}/preview-frames`,
      { sampleFps, maxFrames, dedupThreshold }
    );
  }

  extractVideoFrames(
    projectId: string,
    videoId: string,
    payload: { frameIndices?: number[]; sampleFps?: number; maxFrames?: number; dedupThreshold?: number }
  ): Observable<AnnotationItemSummary[]> {
    return this.http.post<AnnotationItemSummary[]>(
      `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/videos/${encodeURIComponent(videoId)}/extract-frames`,
      payload
    );
  }

  importCurationInbox(projectId: string): Observable<ImportCurationResponse> {
    return this.http.post<ImportCurationResponse>(
      `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/import-curation-inbox`,
      {}
    );
  }

  getItem(projectId: string, itemId: string): Observable<AnnotationItemDetail> {
    return this.http.get<AnnotationItemDetail>(
      `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/items/${encodeURIComponent(itemId)}`
    );
  }

  updateItemAnnotations(
    projectId: string,
    itemId: string,
    payload: UpdateItemAnnotationsRequest
  ): Observable<AnnotationItemDetail> {
    return this.http.put<AnnotationItemDetail>(
      `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/items/${encodeURIComponent(itemId)}/annotations`,
      payload
    );
  }

  deleteItem(projectId: string, itemId: string): Observable<void> {
    return this.http.delete<void>(
      `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/items/${encodeURIComponent(itemId)}`
    );
  }

  preannotateItem(
    projectId: string,
    itemId: string,
    payload: { modelId?: string; runtime?: string; conf?: number; iou?: number; overwrite?: boolean }
  ): Observable<AnnotationItemDetail> {
    return this.http.post<AnnotationItemDetail>(
      `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/items/${encodeURIComponent(itemId)}/preannotate`,
      payload
    );
  }

  preannotateBatch(
    projectId: string,
    payload: { modelId?: string; runtime?: string; conf?: number; iou?: number; overwrite?: boolean }
  ): Observable<PreannotateBatchResponse> {
    return this.http.post<PreannotateBatchResponse>(
      `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/preannotate-batch`,
      payload
    );
  }

  exportYoloZip(projectId: string, includeUnreviewedDrafts = false): Observable<Blob> {
    return this.http.post(
      `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/export-yolo`,
      { onlyHumanReviewed: !includeUnreviewedDrafts, includeUnreviewedDrafts },
      { responseType: 'blob' }
    );
  }

  importYoloZip(projectId: string, file: File): Observable<AnnotationItemSummary[]> {
    const form = new FormData();
    form.append('file', file, file.name);
    return this.http.post<AnnotationItemSummary[]>(
      `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/import-yolo`,
      form
    );
  }
}

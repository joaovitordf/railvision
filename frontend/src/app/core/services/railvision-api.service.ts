import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';
import {
  InspectionDetail,
  InspectionSummary,
  ModelVersion,
  ProcessingJob,
  WagonObservation,
  WagonResult,
} from '../models/inspection.models';

@Injectable({ providedIn: 'root' })
export class RailvisionApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = environment.apiBaseUrl;

  listInspections(): Observable<InspectionSummary[]> {
    return this.http.get<InspectionSummary[]>(`${this.baseUrl}/inspections`);
  }

  getInspection(id: string): Observable<InspectionDetail> {
    return this.http.get<InspectionDetail>(`${this.baseUrl}/inspections/${encodeURIComponent(id)}`);
  }

  createInspection(file: File, title?: string, notes?: string): Observable<InspectionSummary> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    if (title && title.trim().length > 0) {
      formData.append('title', title.trim());
    }
    if (notes && notes.trim().length > 0) {
      formData.append('notes', notes.trim());
    }
    return this.http.post<InspectionSummary>(`${this.baseUrl}/inspections`, formData);
  }

  startProcessing(inspectionId: string): Observable<ProcessingJob> {
    return this.http.post<ProcessingJob>(
      `${this.baseUrl}/inspections/${encodeURIComponent(inspectionId)}/process`,
      {}
    );
  }

  getJob(jobId: string): Observable<ProcessingJob> {
    return this.http.get<ProcessingJob>(`${this.baseUrl}/jobs/${encodeURIComponent(jobId)}`);
  }

  listWagons(inspectionId: string): Observable<WagonResult[]> {
    return this.http.get<WagonResult[]>(
      `${this.baseUrl}/inspections/${encodeURIComponent(inspectionId)}/wagons`
    );
  }

  listObservations(inspectionId: string, wagonId: string): Observable<WagonObservation[]> {
    return this.http.get<WagonObservation[]>(
      `${this.baseUrl}/inspections/${encodeURIComponent(inspectionId)}/wagons/${encodeURIComponent(wagonId)}/observations`
    );
  }

  listModels(): Observable<ModelVersion[]> {
    return this.http.get<ModelVersion[]>(`${this.baseUrl}/models`);
  }

  mediaUrl(inspectionId: string): string {
    return `${this.baseUrl}/inspections/${encodeURIComponent(inspectionId)}/media`;
  }
}

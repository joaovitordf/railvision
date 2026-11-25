import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../../../environments/environment';
import { ModelCheckpoint } from '../models/lab.models';
import {
  RegisterTrainingCheckpointRequest,
  StartTrainingRequest,
  TrainingOverview,
  TrainingRunDetail,
  TrainingRunSummary,
} from '../models/training.models';

@Injectable({ providedIn: 'root' })
export class TrainingApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiBaseUrl}/training`;

  getOverview(): Observable<TrainingOverview> {
    return this.http.get<TrainingOverview>(`${this.baseUrl}/overview`);
  }

  listRuns(): Observable<TrainingRunSummary[]> {
    return this.http.get<TrainingRunSummary[]>(`${this.baseUrl}/runs`);
  }

  getRun(runId: string): Observable<TrainingRunDetail> {
    return this.http.get<TrainingRunDetail>(`${this.baseUrl}/runs/${encodeURIComponent(runId)}`);
  }

  startRun(payload: StartTrainingRequest): Observable<TrainingRunSummary> {
    return this.http.post<TrainingRunSummary>(`${this.baseUrl}/runs`, payload);
  }

  stopRun(runId: string): Observable<TrainingRunSummary> {
    return this.http.post<TrainingRunSummary>(`${this.baseUrl}/runs/${encodeURIComponent(runId)}/stop`, {});
  }

  registerCheckpoint(
    runId: string,
    payload: RegisterTrainingCheckpointRequest = {}
  ): Observable<ModelCheckpoint> {
    return this.http.post<ModelCheckpoint>(
      `${this.baseUrl}/runs/${encodeURIComponent(runId)}/register`,
      payload
    );
  }
}

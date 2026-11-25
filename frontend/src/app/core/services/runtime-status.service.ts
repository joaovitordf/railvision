import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, shareReplay } from 'rxjs';
import { environment } from '../../../environments/environment';
import { ModelCheckpointSummary, RuntimeStatus } from '../models/runtime.models';

@Injectable({ providedIn: 'root' })
export class RuntimeStatusService {
  private readonly http = inject(HttpClient);
  private readonly statusUrl = `${environment.apiBaseUrl}/vision-lab/status`;
  private readonly checkpointsUrl = `${environment.apiBaseUrl}/models/checkpoints`;
  private cached$?: Observable<RuntimeStatus>;

  getStatus(forceRefresh = false): Observable<RuntimeStatus> {
    if (forceRefresh || !this.cached$) {
      this.cached$ = this.http.get<RuntimeStatus>(this.statusUrl).pipe(shareReplay(1));
    }
    return this.cached$;
  }

  listCheckpoints(): Observable<ModelCheckpointSummary[]> {
    return this.http.get<ModelCheckpointSummary[]>(this.checkpointsUrl);
  }
}

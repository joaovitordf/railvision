import { Injectable, inject } from '@angular/core';
import { Observable, catchError, forkJoin, map, of, switchMap } from 'rxjs';
import { InspectionDetail, InspectionSummary, WagonObservation, WagonResult } from '../models/inspection.models';
import { RailvisionApiService } from './railvision-api.service';

export interface InspectionReportBundle {
  detail: InspectionDetail;
  wagons: WagonResult[];
  observationsByWagonId: Record<string, WagonObservation[]>;
}

@Injectable({ providedIn: 'root' })
export class InspectionReportExportService {
  private readonly api = inject(RailvisionApiService);

  fetchBundle(inspectionId: string): Observable<InspectionReportBundle> {
    return forkJoin({
      detail: this.api.getInspection(inspectionId),
      wagons: this.api.listWagons(inspectionId),
    }).pipe(
      switchMap(({ detail, wagons }) => {
        if (wagons.length === 0) {
          return of({
            detail,
            wagons,
            observationsByWagonId: {} as Record<string, WagonObservation[]>,
          });
        }
        return forkJoin(
          wagons.map((wagon) =>
            this.api.listObservations(inspectionId, wagon.id).pipe(
              catchError(() => of([] as WagonObservation[])),
              map((observations) => [wagon.id, observations] as const)
            )
          )
        ).pipe(
          map((entries) => ({
            detail,
            wagons,
            observationsByWagonId: Object.fromEntries(entries) as Record<string, WagonObservation[]>,
          }))
        );
      })
    );
  }

  downloadJson(bundle: InspectionReportBundle, filenamePrefix?: string): void {
    const { detail, wagons, observationsByWagonId } = bundle;
    const payload = {
      generatedAt: new Date().toISOString(),
      isExampleReport: wagons.some((w) => w.isExample) || Boolean(detail.latestJob?.isDemoSimulation),
      inspection: detail,
      wagons: wagons.map((wagon) => ({
        ...wagon,
        observations: observationsByWagonId[wagon.id] ?? [],
      })),
    };
    this.triggerDownload(
      new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }),
      `${filenamePrefix ?? `railvision-${detail.id.slice(0, 8)}`}.json`
    );
  }

  downloadCsv(bundle: InspectionReportBundle, filenamePrefix?: string): void {
    const { detail, wagons, observationsByWagonId } = bundle;
    const header = [
      'inspection_id',
      'inspection_title',
      'wagon_index',
      'track_id',
      'loading_level',
      'loading_confidence',
      'material_class',
      'material_confidence',
      'observations_count',
      'first_timestamp_ms',
      'last_timestamp_ms',
      'quality_flags',
    ];
    const rows = wagons.map((wagon) => {
      const obs = observationsByWagonId[wagon.id] ?? [];
      const firstTs = obs[0]?.frameTimestampMs ?? '';
      const lastTs = obs.length > 0 ? (obs[obs.length - 1]?.frameTimestampMs ?? '') : '';
      return [
        detail.id,
        detail.title,
        wagon.wagonIdentity.sequenceIndex,
        wagon.wagonIdentity.trackId,
        wagon.loadingEstimate.level,
        wagon.loadingEstimate.confidence ?? '',
        wagon.material?.materialClass ?? '',
        wagon.material?.confidence ?? '',
        obs.length,
        firstTs,
        lastTs,
        (wagon.qualityFlags ?? []).join(';'),
      ]
        .map((cell) => `"${String(cell ?? '').replaceAll('"', '""')}"`)
        .join(',');
    });
    const csv = [header.join(','), ...rows].join('\r\n');
    this.triggerDownload(
      new Blob([csv], { type: 'text/csv;charset=utf-8' }),
      `${filenamePrefix ?? `railvision-${detail.id.slice(0, 8)}`}.csv`
    );
  }

  exportSummaryById(
    item: Pick<InspectionSummary, 'id'>,
    format: 'json' | 'csv',
    callbacks: { onSuccess: () => void; onError: (err: unknown) => void }
  ): void {
    this.fetchBundle(item.id).subscribe({
      next: (bundle) => {
        if (format === 'json') {
          this.downloadJson(bundle);
        } else {
          this.downloadCsv(bundle);
        }
        callbacks.onSuccess();
      },
      error: (err) => callbacks.onError(err),
    });
  }

  private triggerDownload(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
  }
}

import { Component, inject, input, signal } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { InspectionDetail, WagonObservation, WagonResult } from '../../../core/models/inspection.models';
import { extractApiErrorMessage } from '../../../core/services/api-error.util';
import { InspectionReportExportService } from '../../../core/services/inspection-report-export.service';
import { NotificationService } from '../../../core/services/notification.service';

@Component({
  selector: 'rv-inspection-report-tab',
  standalone: true,
  imports: [MatIconModule],
  template: `
    <section class="rv-card">
      <div class="rv-card__header">
        <div>
          <h3 class="rv-card__title">Relatório consolidado da inspeção</h3>
          <p class="rv-card__subtitle">
            Exporte todos os dados consolidados desta inspeção em JSON completo ou CSV tabular.
          </p>
        </div>

        <div class="rv-header-actions">
          <button type="button" class="rv-btn rv-btn--primary" [disabled]="exporting()" (click)="exportBundle('json')">
            <mat-icon>download</mat-icon>
            <span>Exportar JSON</span>
          </button>
          <button type="button" class="rv-btn rv-btn--secondary" [disabled]="exporting()" (click)="exportBundle('csv')">
            <mat-icon>table_view</mat-icon>
            <span>Exportar CSV</span>
          </button>
        </div>
      </div>

      <div class="rv-kpi-grid" style="margin-bottom: 18px;">
        <div class="rv-kpi-card">
          <div class="rv-kpi-card__body">
            <span class="rv-kpi-card__label">Status da inspeção</span>
            <span class="rv-kpi-card__value" style="font-size: 1.35rem;">{{ detail().status }}</span>
          </div>
        </div>
        <div class="rv-kpi-card">
          <div class="rv-kpi-card__body">
            <span class="rv-kpi-card__label">Vagões consolidados</span>
            <span class="rv-kpi-card__value">{{ wagons().length }}</span>
          </div>
        </div>
        <div class="rv-kpi-card">
          <div class="rv-kpi-card__body">
            <span class="rv-kpi-card__label">Observações totais</span>
            <span class="rv-kpi-card__value">{{ totalObservations() }}</span>
          </div>
        </div>
      </div>

      <p style="margin: 0; font-size: 0.85rem; color: var(--rv-text-secondary);">
        Os arquivos exportados refletem exatamente os resultados persistidos no PostgreSQL para o job concluído mais recente,
        incluindo indicação de registros históricos ou itens marcados para revisão.
      </p>
    </section>
  `,
})
export class InspectionReportTabComponent {
  private readonly exporter = inject(InspectionReportExportService);
  private readonly notifications = inject(NotificationService);

  readonly detail = input.required<InspectionDetail>();
  readonly wagons = input<WagonResult[]>([]);
  readonly observationsByWagon = input<Record<string, WagonObservation[]>>({});

  protected readonly exporting = signal<boolean>(false);

  protected totalObservations(): number {
    return this.wagons().reduce((acc, w) => acc + (w.consolidation?.observationsCount ?? 0), 0);
  }

  protected exportBundle(format: 'json' | 'csv'): void {
    const det = this.detail();
    const cached = this.observationsByWagon();
    const allCached = this.wagons().every((w) => Boolean(cached[w.id]));
    if (allCached) {
      if (format === 'json') {
        this.exporter.downloadJson({ detail: det, wagons: this.wagons(), observationsByWagonId: cached });
      } else {
        this.exporter.downloadCsv({ detail: det, wagons: this.wagons(), observationsByWagonId: cached });
      }
      this.notifications.success(`Relatório ${format.toUpperCase()} exportado.`);
      return;
    }

    this.exporting.set(true);
    this.exporter.exportSummaryById(det, format, {
      onSuccess: () => {
        this.exporting.set(false);
        this.notifications.success(`Relatório ${format.toUpperCase()} exportado.`);
      },
      onError: (err) => {
        this.exporting.set(false);
        this.notifications.error(
          extractApiErrorMessage(err, `Falha ao exportar relatório ${format.toUpperCase()}.`)
        );
      },
    });
  }
}

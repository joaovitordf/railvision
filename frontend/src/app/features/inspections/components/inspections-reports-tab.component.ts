import { DatePipe } from '@angular/common';
import { Component, computed, inject, input, output, signal } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { RouterLink } from '@angular/router';
import { InspectionSummary } from '../../../core/models/inspection.models';
import { extractApiErrorMessage } from '../../../core/services/api-error.util';
import { InspectionReportExportService } from '../../../core/services/inspection-report-export.service';
import { NotificationService } from '../../../core/services/notification.service';
import { EmptyStateComponent } from '../../../shared/components/empty-state/empty-state.component';
import { StatusBadgeComponent } from '../../../shared/components/status-badge/status-badge.component';
import { FormatDurationPipe } from '../../../shared/pipes/format-duration.pipe';

@Component({
  selector: 'rv-inspections-reports-tab',
  standalone: true,
  imports: [RouterLink, DatePipe, MatIconModule, StatusBadgeComponent, EmptyStateComponent, FormatDurationPipe],
  template: `
    <section class="rv-card rv-card--table">
      <div class="rv-card__header">
        <div>
          <h3 class="rv-card__title">Relatórios de inspeções concluídas</h3>
          <p class="rv-card__subtitle">
            Exporte dados estruturados em JSON ou CSV com metadados, tracks e observações reais.
          </p>
        </div>
        <button type="button" class="rv-btn rv-btn--secondary rv-btn--sm" (click)="refreshRequested.emit()">
          <mat-icon>refresh</mat-icon>
          <span>Atualizar</span>
        </button>
      </div>

      @if (completedInspections().length === 0) {
        <rv-empty-state
          icon="summarize"
          title="Nenhuma inspeção concluída"
          description="Conclua o processamento de uma inspeção para liberar os relatórios e exportações."
        />
      } @else {
        <div class="rv-table-wrap">
          <table class="rv-table">
            <thead>
              <tr>
                <th>Inspeção</th>
                <th>Arquivo</th>
                <th>Duração</th>
                <th>Vagões</th>
                <th>Status</th>
                <th>Atualizada em</th>
                <th style="text-align: right;">Exportação</th>
              </tr>
            </thead>
            <tbody>
              @for (item of completedInspections(); track item.id) {
                <tr>
                  <td>
                    <a [routerLink]="['/inspections', item.id, 'report']" style="color: var(--rv-accent); font-weight: 600; text-decoration: none;">
                      {{ item.title }}
                    </a>
                  </td>
                  <td class="rv-mono" style="color: var(--rv-text-secondary);">{{ item.originalFilename }}</td>
                  <td class="rv-mono">{{ item.durationMs | formatDuration }}</td>
                  <td>{{ item.wagonCount ?? 0 }}</td>
                  <td><rv-status-badge [status]="item.status" /></td>
                  <td style="color: var(--rv-text-secondary);">{{ item.updatedAt | date: 'dd/MM/yyyy HH:mm' }}</td>
                  <td style="text-align: right;">
                    <div style="display: inline-flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end;">
                      <button
                        type="button"
                        class="rv-btn rv-btn--primary rv-btn--sm"
                        [disabled]="exportingId() === item.id"
                        (click)="exportReport(item, 'json')"
                      >
                        <mat-icon>download</mat-icon>
                        <span>JSON</span>
                      </button>
                      <button
                        type="button"
                        class="rv-btn rv-btn--secondary rv-btn--sm"
                        [disabled]="exportingId() === item.id"
                        (click)="exportReport(item, 'csv')"
                      >
                        <mat-icon>table_view</mat-icon>
                        <span>CSV</span>
                      </button>
                    </div>
                  </td>
                </tr>
              }
            </tbody>
          </table>
        </div>
      }
    </section>
  `,
})
export class InspectionsReportsTabComponent {
  private readonly exporter = inject(InspectionReportExportService);
  private readonly notifications = inject(NotificationService);

  readonly inspections = input<InspectionSummary[]>([]);
  readonly refreshRequested = output<void>();

  protected readonly exportingId = signal<string | null>(null);
  protected readonly completedInspections = computed(() =>
    this.inspections().filter((i) => i.status === 'COMPLETED')
  );

  protected exportReport(item: InspectionSummary, format: 'json' | 'csv'): void {
    this.exportingId.set(item.id);
    this.exporter.exportSummaryById(item, format, {
      onSuccess: () => {
        this.exportingId.set(null);
        this.notifications.success(`Relatório ${format.toUpperCase()} exportado.`);
      },
      onError: (err) => {
        this.exportingId.set(null);
        this.notifications.error(
          extractApiErrorMessage(err, `Falha ao exportar relatório ${format.toUpperCase()}.`)
        );
      },
    });
  }
}

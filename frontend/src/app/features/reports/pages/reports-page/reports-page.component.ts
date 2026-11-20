import { DatePipe } from '@angular/common';
import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { RouterLink } from '@angular/router';
import { forkJoin } from 'rxjs';
import { InspectionSummary } from '../../../../core/models/inspection.models';
import { extractApiErrorMessage } from '../../../../core/services/api-error.util';
import { NotificationService } from '../../../../core/services/notification.service';
import { RailvisionApiService } from '../../../../core/services/railvision-api.service';
import { EmptyStateComponent } from '../../../../shared/components/empty-state/empty-state.component';
import { StatusBadgeComponent } from '../../../../shared/components/status-badge/status-badge.component';
import { FormatBytesPipe } from '../../../../shared/pipes/format-bytes.pipe';

@Component({
  selector: 'rv-reports-page',
  standalone: true,
  imports: [
    RouterLink,
    DatePipe,
    MatButtonModule,
    MatIconModule,
    MatProgressBarModule,
    StatusBadgeComponent,
    EmptyStateComponent,
    FormatBytesPipe,
  ],
  template: `
    <div class="rv-page">
      <section class="rv-card">
        <div class="rv-card__header">
          <div>
            <h2 class="rv-card__title">Relatórios de inspeção</h2>
            <p class="rv-card__subtitle">
              Exporte o resumo estruturado em JSON com metadados, tracks e observações reais da inspeção.
            </p>
          </div>
          <button type="button" mat-stroked-button (click)="load()" [disabled]="loading()">
            <mat-icon>refresh</mat-icon>
            Atualizar
          </button>
        </div>

        @if (loading()) {
          <mat-progress-bar mode="indeterminate" />
        }

        @if (errorMessage()) {
          <div class="rv-banner rv-banner--error" role="alert">
            <mat-icon aria-hidden="true">error_outline</mat-icon>
            <div>{{ errorMessage() }}</div>
          </div>
        } @else if (completedInspections().length === 0 && !loading()) {
          <rv-empty-state
            icon="summarize"
            title="Nenhuma inspeção concluída"
            description="Conclua o processamento de pelo menos uma inspeção para gerar relatórios."
          />
        } @else if (completedInspections().length > 0) {
          <div class="rv-table-wrap">
            <table class="rv-table">
              <thead>
                <tr>
                  <th>Inspeção</th>
                  <th>Arquivo</th>
                  <th>Tamanho</th>
                  <th>Status</th>
                  <th>Atualizada em</th>
                  <th>Exportação</th>
                </tr>
              </thead>
              <tbody>
                @for (item of completedInspections(); track item.id) {
                  <tr>
                    <td>
                      <a [routerLink]="['/inspections', item.id]" style="color: var(--rv-accent); font-weight: 600;">
                        {{ item.title }}
                      </a>
                    </td>
                    <td class="rv-mono">{{ item.originalFilename }}</td>
                    <td>{{ item.fileSizeBytes | formatBytes }}</td>
                    <td><rv-status-badge [status]="item.status" /></td>
                    <td>{{ item.updatedAt | date: 'dd/MM/yyyy HH:mm' }}</td>
                    <td>
                      <button
                        type="button"
                        mat-flat-button
                        color="primary"
                        [disabled]="exportingId() === item.id"
                        (click)="exportJson(item)"
                      >
                        <mat-icon>download</mat-icon>
                        Exportar JSON
                      </button>
                    </td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
        }
      </section>
    </div>
  `,
})
export class ReportsPageComponent implements OnInit {
  private readonly api = inject(RailvisionApiService);
  private readonly notifications = inject(NotificationService);

  protected readonly loading = signal<boolean>(false);
  protected readonly exportingId = signal<string | null>(null);
  protected readonly errorMessage = signal<string | null>(null);
  protected readonly inspections = signal<InspectionSummary[]>([]);

  protected readonly completedInspections = computed(() =>
    this.inspections().filter((i) => i.status === 'COMPLETED')
  );

  ngOnInit(): void {
    this.load();
  }

  protected load(): void {
    this.loading.set(true);
    this.errorMessage.set(null);
    this.api.listInspections().subscribe({
      next: (items) => {
        this.inspections.set(items);
        this.loading.set(false);
      },
      error: (err) => {
        this.loading.set(false);
        this.errorMessage.set(
          extractApiErrorMessage(err, 'Não foi possível carregar os relatórios.')
        );
      },
    });
  }

  protected exportJson(item: InspectionSummary): void {
    this.exportingId.set(item.id);
    forkJoin({
      detail: this.api.getInspection(item.id),
      wagons: this.api.listWagons(item.id),
    }).subscribe({
      next: ({ detail, wagons }) => {
        const payload = {
          generatedAt: new Date().toISOString(),
          isExampleReport: wagons.some((w) => w.isExample) || Boolean(detail.latestJob?.isDemoSimulation),
          inspection: detail,
          wagons,
        };
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = `railvision-${item.id.slice(0, 8)}.json`;
        anchor.click();
        URL.revokeObjectURL(url);
        this.exportingId.set(null);
        this.notifications.success('Relatório JSON exportado.');
      },
      error: (err) => {
        this.exportingId.set(null);
        this.notifications.error(extractApiErrorMessage(err, 'Falha ao exportar relatório.'));
      },
    });
  }
}

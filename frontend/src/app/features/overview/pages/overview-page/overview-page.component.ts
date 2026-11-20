import { DatePipe } from '@angular/common';
import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { RouterLink } from '@angular/router';
import { catchError, forkJoin, of } from 'rxjs';
import { InspectionSummary, ModelVersion } from '../../../../core/models/inspection.models';
import { extractApiErrorMessage } from '../../../../core/services/api-error.util';
import { RailvisionApiService } from '../../../../core/services/railvision-api.service';
import { EmptyStateComponent } from '../../../../shared/components/empty-state/empty-state.component';
import { StatusBadgeComponent } from '../../../../shared/components/status-badge/status-badge.component';
import { FormatBytesPipe } from '../../../../shared/pipes/format-bytes.pipe';
import { RuntimeStatus } from '../../../onnx-lab/models/onnx-lab.models';
import { OnnxLabApiService } from '../../../onnx-lab/services/onnx-lab-api.service';

@Component({
  selector: 'rv-overview-page',
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
      @if (runtimeStatus(); as st) {
        <section class="rv-banner" [class.rv-banner--info]="st.available" [class.rv-banner--warning]="!st.available">
          <mat-icon aria-hidden="true">{{ st.deviceType === 'GPU' ? 'memory' : 'info' }}</mat-icon>
          <div style="display: flex; flex-direction: column; gap: 4px;">
            <strong>
              Backend Python + FastAPI · Runtime ativo: {{ st.deviceType }} ({{ st.activeExecutionProvider }})
            </strong>
            <span>
              Modelo de detecção <span class="rv-mono">{{ st.modelName }}</span>
              (SHA <span class="rv-mono">{{ st.modelSha256.slice(0, 12) }}</span>) + tracking ByteTrack em resolução original via PyAV.
              Classificação de material e nível de carregamento permanecem como
              <span class="rv-mono">not_evaluated</span> até o treinamento dos modelos dedicados.
            </span>
          </div>
        </section>
      }

      @if (loading()) {
        <mat-progress-bar mode="indeterminate" />
      }

      @if (errorMessage()) {
        <div class="rv-banner rv-banner--error" role="alert">
          <mat-icon aria-hidden="true">error_outline</mat-icon>
          <div>{{ errorMessage() }}</div>
        </div>
      }

      <section class="rv-kpi-grid" aria-label="Resumo operacional">
        <div class="rv-kpi-card">
          <span class="rv-kpi-card__label">Total de inspeções</span>
          <span class="rv-kpi-card__value">{{ totalCount() }}</span>
        </div>
        <div class="rv-kpi-card">
          <span class="rv-kpi-card__label">Concluídas</span>
          <span class="rv-kpi-card__value">{{ completedCount() }}</span>
        </div>
        <div class="rv-kpi-card">
          <span class="rv-kpi-card__label">Em fila ou processamento</span>
          <span class="rv-kpi-card__value">{{ activeCount() }}</span>
        </div>
        <div class="rv-kpi-card">
          <span class="rv-kpi-card__label">Com falha</span>
          <span class="rv-kpi-card__value">{{ failedCount() }}</span>
        </div>
      </section>

      <section class="rv-card">
        <div class="rv-card__header">
          <div>
            <h2 class="rv-card__title">Inspeções recentes</h2>
            <p class="rv-card__subtitle">Últimos vídeos ferroviários registrados nesta instância local.</p>
          </div>
          <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <button type="button" mat-stroked-button (click)="load()">
              <mat-icon>refresh</mat-icon>
              Atualizar
            </button>
            <a mat-flat-button color="primary" routerLink="/inspections">
              <mat-icon>upload</mat-icon>
              Nova inspeção
            </a>
          </div>
        </div>

        @if (recentInspections().length === 0 && !loading()) {
          <rv-empty-state
            icon="train"
            title="Nenhuma inspeção registrada"
            description="Envie um vídeo ferroviário na aba Inspeções para iniciar uma análise local com PyTorch CUDA e ByteTrack."
          />
        } @else if (recentInspections().length > 0) {
          <div class="rv-table-wrap">
            <table class="rv-table">
              <thead>
                <tr>
                  <th>Título</th>
                  <th>Arquivo</th>
                  <th>Tamanho</th>
                  <th>Status</th>
                  <th>Criado em</th>
                  <th>Ação</th>
                </tr>
              </thead>
              <tbody>
                @for (item of recentInspections(); track item.id) {
                  <tr>
                    <td><strong>{{ item.title }}</strong></td>
                    <td class="rv-mono">{{ item.originalFilename }}</td>
                    <td>{{ item.fileSizeBytes | formatBytes }}</td>
                    <td><rv-status-badge [status]="item.status" /></td>
                    <td>{{ item.createdAt | date: 'dd/MM/yyyy HH:mm' }}</td>
                    <td>
                      <a mat-button color="primary" [routerLink]="['/inspections', item.id]">
                        Abrir detalhe
                      </a>
                    </td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
        }
      </section>

      <section class="rv-card">
        <div class="rv-card__header">
          <div>
            <h2 class="rv-card__title">Modelos registrados</h2>
            <p class="rv-card__subtitle">Metadados persistidos após execução de jobs no backend.</p>
          </div>
        </div>

        @if (models().length === 0) {
          <rv-empty-state
            icon="model_training"
            title="Nenhum modelo registrado ainda"
            description="Os registros aparecem após processar a primeira inspeção."
          />
        } @else {
          <div class="rv-table-wrap">
            <table class="rv-table">
              <thead>
                <tr>
                  <th>Tarefa</th>
                  <th>Nome</th>
                  <th>Versão</th>
                  <th>Runtime</th>
                  <th>Tipo</th>
                </tr>
              </thead>
              <tbody>
                @for (m of models(); track m.id) {
                  <tr>
                    <td class="rv-mono">{{ m.task }}</td>
                    <td>{{ m.name }}</td>
                    <td class="rv-mono">{{ m.version }}</td>
                    <td class="rv-mono">{{ m.runtime }}</td>
                    <td>
                      @if (m.isDemoPlaceholder) {
                        <span class="rv-badge rv-badge--warning">Placeholder de demonstração</span>
                      } @else {
                        <span class="rv-badge rv-badge--success">Modelo ativo</span>
                      }
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
export class OverviewPageComponent implements OnInit {
  private readonly api = inject(RailvisionApiService);
  private readonly labApi = inject(OnnxLabApiService);

  protected readonly loading = signal<boolean>(false);
  protected readonly errorMessage = signal<string | null>(null);
  protected readonly inspections = signal<InspectionSummary[]>([]);
  protected readonly models = signal<ModelVersion[]>([]);
  protected readonly runtimeStatus = signal<RuntimeStatus | null>(null);

  protected readonly totalCount = computed(() => this.inspections().length);
  protected readonly completedCount = computed(
    () => this.inspections().filter((i) => i.status === 'COMPLETED').length
  );
  protected readonly activeCount = computed(
    () => this.inspections().filter((i) => i.status === 'QUEUED' || i.status === 'PROCESSING').length
  );
  protected readonly failedCount = computed(
    () => this.inspections().filter((i) => i.status === 'FAILED').length
  );
  protected readonly recentInspections = computed(() => this.inspections().slice(0, 6));

  ngOnInit(): void {
    this.load();
  }

  protected load(): void {
    this.loading.set(true);
    this.errorMessage.set(null);
    forkJoin({
      inspections: this.api.listInspections(),
      models: this.api.listModels(),
      runtimeStatus: this.labApi.getStatus().pipe(catchError(() => of(null))),
    }).subscribe({
      next: ({ inspections, models, runtimeStatus }) => {
        this.inspections.set(inspections);
        this.models.set(models);
        this.runtimeStatus.set(runtimeStatus);
        this.loading.set(false);
      },
      error: (err) => {
        this.errorMessage.set(
          extractApiErrorMessage(err, 'Não foi possível carregar os dados do painel.')
        );
        this.loading.set(false);
      },
    });
  }
}

import { DecimalPipe } from '@angular/common';
import { Component, input, output } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { ModelVersion } from '../../../core/models/inspection.models';
import { EmptyStateComponent } from '../../../shared/components/empty-state/empty-state.component';
import { ModelCheckpoint, RuntimeStatus } from '../models/lab.models';

@Component({
  selector: 'rv-lab-models-registry',
  standalone: true,
  imports: [DecimalPipe, MatIconModule, EmptyStateComponent],
  template: `
    <div class="rv-page">
      <section class="rv-card">
        <div class="rv-card__header">
          <div>
            <h3 class="rv-card__title">Diagnóstico de Hardware e Runtime</h3>
            <p class="rv-card__subtitle">
              Dispositivo verificado explicitamente em tempo de execução (sem fallback silencioso para CPU).
            </p>
          </div>
          @if (status(); as st) {
            <span class="rv-badge" [class.rv-badge--success]="st.available" [class.rv-badge--error]="!st.available">
              <span class="rv-badge__dot" aria-hidden="true"></span>
              {{ st.available ? st.deviceType + ' (' + st.activeExecutionProvider + ')' : 'Indisponível' }}
            </span>
          }
        </div>

        @if (status(); as st) {
          <div class="rv-kpi-grid">
            <div class="rv-kpi-card">
              <div class="rv-kpi-card__body">
                <span class="rv-kpi-card__label">Modelo ativo</span>
                <span class="rv-mono" style="font-weight: 600;">{{ st.modelName }}</span>
                <span class="rv-kpi-card__meta rv-mono">SHA {{ st.modelSha256.slice(0, 12) }}</span>
              </div>
            </div>
            <div class="rv-kpi-card">
              <div class="rv-kpi-card__body">
                <span class="rv-kpi-card__label">Dispositivo / Provedor</span>
                <span class="rv-kpi-card__value" style="font-size: 1.4rem;">{{ st.deviceType }}</span>
                <span class="rv-kpi-card__meta rv-mono">{{ st.activeExecutionProvider }}</span>
              </div>
            </div>
            <div class="rv-kpi-card">
              <div class="rv-kpi-card__body">
                <span class="rv-kpi-card__label">Tensor entrada / saída</span>
                <span class="rv-mono" style="font-weight: 600;">{{ st.inputShape.join(' × ') }}</span>
                <span class="rv-kpi-card__meta rv-mono">Saída {{ st.outputShape.join(' × ') }}</span>
              </div>
            </div>
            <div class="rv-kpi-card">
              <div class="rv-kpi-card__body">
                <span class="rv-kpi-card__label">Tracking</span>
                <span style="font-weight: 600;">
                  {{ st.trackingAlgorithms?.includes('bytetrack') ? 'ByteTrack disponível' : 'Somente detecção' }}
                </span>
                <span class="rv-kpi-card__meta">Habilitável por execução em vídeo</span>
              </div>
            </div>
          </div>

          <p style="margin: 14px 0 0; font-size: 0.84rem; color: var(--rv-text-secondary);">
            {{ st.deviceDiagnostic }}
          </p>
        }
      </section>

      <section class="rv-card rv-card--table">
        <div class="rv-card__header">
          <div>
            <h3 class="rv-card__title">Model Registry — Checkpoints Versionados</h3>
            <p class="rv-card__subtitle">
              Catálogo oficial de checkpoints de detecção (<span class="rv-mono">artifacts/models/registry.json</span>).
            </p>
          </div>
        </div>

        @if (checkpoints().length === 0) {
          <rv-empty-state
            icon="memory"
            title="Nenhum checkpoint encontrado"
            description="Verifique artifacts/models/registry.json."
          />
        } @else {
          <div class="rv-table-wrap">
            <table class="rv-table">
              <thead>
                <tr>
                  <th>ID / Experimento</th>
                  <th>Arquitetura</th>
                  <th>Status</th>
                  <th>Val mAP50</th>
                  <th>Test mAP50</th>
                  <th>SHA-256 (PyTorch)</th>
                  <th>Ação</th>
                </tr>
              </thead>
              <tbody>
                @for (cp of checkpoints(); track cp.id) {
                  <tr>
                    <td>
                      <strong class="rv-mono">{{ cp.id }}</strong>
                      <div class="rv-mono" style="font-size: 0.76rem; color: var(--rv-text-muted);">{{ cp.sourceExperiment }}</div>
                    </td>
                    <td class="rv-mono">{{ cp.architecture }} ({{ cp.inputSize }}px)</td>
                    <td>
                      <span
                        class="rv-badge"
                        [class.rv-badge--success]="cp.status === 'active'"
                        [class.rv-badge--info]="cp.status === 'candidate'"
                        [class.rv-badge--neutral]="cp.status === 'archived'"
                      >
                        <span class="rv-badge__dot" aria-hidden="true"></span>
                        {{ cp.status }}
                      </span>
                    </td>
                    <td class="rv-mono">{{ cp.valMetrics['map50'] != null ? (cp.valMetrics['map50'] | number: '1.3-3') : '—' }}</td>
                    <td class="rv-mono">{{ cp.testMetrics['map50'] != null ? (cp.testMetrics['map50'] | number: '1.4-4') : '—' }}</td>
                    <td class="rv-mono">{{ cp.pytorchWeightsSha256 ? cp.pytorchWeightsSha256.slice(0, 14) : '—' }}</td>
                    <td>
                      <button
                        type="button"
                        class="rv-btn rv-btn--secondary rv-btn--sm"
                        (click)="selectForInference.emit(cp.id)"
                      >
                        <mat-icon>play_arrow</mat-icon>
                        <span>Testar modelo</span>
                      </button>
                    </td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
        }
      </section>

      <section class="rv-card rv-card--table">
        <div class="rv-card__header">
          <div>
            <h3 class="rv-card__title">Histórico persistido no PostgreSQL (<span class="rv-mono">model_versions</span>)</h3>
            <p class="rv-card__subtitle">Versões e provedores gravados durante jobs assíncronos de inspeção.</p>
          </div>
        </div>

        @if (dbModels().length === 0) {
          <rv-empty-state
            icon="storage"
            title="Nenhum registro persistido ainda"
            description="Os metadados aparecem após a execução de inspeções no backend."
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
                  <th>SHA-256</th>
                </tr>
              </thead>
              <tbody>
                @for (m of dbModels(); track m.id) {
                  <tr>
                    <td class="rv-mono">{{ m.task }}</td>
                    <td><strong>{{ m.name }}</strong></td>
                    <td class="rv-mono">{{ m.version }}</td>
                    <td class="rv-mono">{{ m.runtime }}</td>
                    <td class="rv-mono">{{ m.weightsSha256 ? m.weightsSha256.slice(0, 12) : '—' }}</td>
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
export class LabModelsRegistryComponent {
  readonly status = input<RuntimeStatus | null>(null);
  readonly checkpoints = input<ModelCheckpoint[]>([]);
  readonly dbModels = input<ModelVersion[]>([]);
  readonly selectForInference = output<string>();
}

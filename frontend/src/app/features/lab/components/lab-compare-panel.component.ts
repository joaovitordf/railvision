import { DecimalPipe } from '@angular/common';
import { Component, computed, input, output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { EmptyStateComponent } from '../../../shared/components/empty-state/empty-state.component';
import { RunComparison, RunListItem } from '../models/lab.models';

@Component({
  selector: 'rv-lab-compare-panel',
  standalone: true,
  imports: [
    DecimalPipe,
    FormsModule,
    MatIconModule,
    EmptyStateComponent,
  ],
  template: `
    <section class="rv-card">
      <div class="rv-card__header">
        <div>
          <h3 class="rv-card__title">Comparação A/B de execuções</h3>
          <p class="rv-card__subtitle">
            Compare dois runs concluídos lado a lado (modelos, thresholds, FPS e delta de detecções).
          </p>
        </div>
      </div>

      @if (completedRuns().length < 2) {
        <rv-empty-state
          icon="compare_arrows"
          title="São necessárias pelo menos duas execuções concluídas"
          description="Execute dois testes na aba Inferência (ex.: dois checkpoints ou GPU vs CPU) para comparar os resultados aqui."
        />
      } @else {
        <div class="rv-form-grid" style="margin-bottom: 18px;">
          <div class="rv-field">
            <label class="rv-label" for="cmp-run-a">Run A (baseline)</label>
            <select
              id="cmp-run-a"
              class="rv-select"
              [ngModel]="runA()"
              (ngModelChange)="runA.set($event)"
              data-testid="compare-run-a"
            >
              <option [ngValue]="null">Selecione uma execução...</option>
              @for (item of completedRuns(); track item.runId) {
                <option [ngValue]="item.runId">
                  {{ item.originalFilename }} · {{ item.modelName }} ({{ item.totalDetections }} det.)
                </option>
              }
            </select>
          </div>

          <div class="rv-field">
            <label class="rv-label" for="cmp-run-b">Run B (candidato)</label>
            <select
              id="cmp-run-b"
              class="rv-select"
              [ngModel]="runB()"
              (ngModelChange)="runB.set($event)"
              data-testid="compare-run-b"
            >
              <option [ngValue]="null">Selecione uma execução...</option>
              @for (item of completedRuns(); track item.runId) {
                <option [ngValue]="item.runId">
                  {{ item.originalFilename }} · {{ item.modelName }} ({{ item.totalDetections }} det.)
                </option>
              }
            </select>
          </div>

          <div style="display: flex;">
            <button
              type="button"
              class="rv-btn rv-btn--primary"
              data-testid="compare-runs-btn"
              [disabled]="!runA() || !runB() || comparing()"
              (click)="compareRequested.emit({ runA: runA()!, runB: runB()! })"
            >
              <mat-icon>compare_arrows</mat-icon>
              <span>{{ comparing() ? 'Comparando...' : 'Comparar execuções' }}</span>
            </button>
          </div>
        </div>

        @if (comparison(); as cmp) {
          @if (!cmp.comparable && cmp.comparabilityWarnings.length > 0) {
            <div class="rv-banner rv-banner--warning" role="status" style="margin-bottom: 16px;" data-testid="compare-warnings-banner">
              <mat-icon aria-hidden="true">warning</mat-icon>
              <div>
                <strong>Atenção à comparabilidade entre execuções:</strong>
                <ul style="margin: 4px 0 0; padding-left: 18px;">
                  @for (warn of cmp.comparabilityWarnings; track warn) {
                    <li>{{ warn }}</li>
                  }
                </ul>
              </div>
            </div>
          }

          @if (cmp.runA.mediaType === 'IMAGE' && cmp.runB.mediaType === 'IMAGE') {
            <div class="rv-two-col-grid" style="margin-bottom: 18px;">
              <div>
                <div style="font-size: 0.84rem; font-weight: 600; margin-bottom: 8px;">
                  Run A · <span class="rv-mono">{{ cmp.runA.modelName }}</span> ({{ cmp.runA.totalDetections }} det.)
                </div>
                <img
                  [src]="'/api/v1/vision-lab/runs/' + cmp.runA.runId + '/files/annotated'"
                  alt="Resultado anotado Run A"
                  style="width: 100%; max-height: 360px; object-fit: contain; border-radius: var(--rv-radius-md); border: 1px solid var(--rv-border); background: #060b13;"
                />
              </div>
              <div>
                <div style="font-size: 0.84rem; font-weight: 600; margin-bottom: 8px;">
                  Run B · <span class="rv-mono">{{ cmp.runB.modelName }}</span> ({{ cmp.runB.totalDetections }} det.)
                </div>
                <img
                  [src]="'/api/v1/vision-lab/runs/' + cmp.runB.runId + '/files/annotated'"
                  alt="Resultado anotado Run B"
                  style="width: 100%; max-height: 360px; object-fit: contain; border-radius: var(--rv-radius-md); border: 1px solid var(--rv-border); background: #060b13;"
                />
              </div>
            </div>
          }

          <div class="rv-card rv-card--table" style="box-shadow: none;">
            <div class="rv-table-wrap">
              <table class="rv-table">
                <thead>
                  <tr>
                    <th>Métrica / Parâmetro</th>
                    <th>Run A ({{ cmp.runA.modelName }})</th>
                    <th>Run B ({{ cmp.runB.modelName }})</th>
                    <th>Delta (B - A)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Arquivo / Origem</td>
                    <td class="rv-mono">{{ cmp.runA.originalFilename }} ({{ cmp.runA.sourceResolution ?? '—' }})</td>
                    <td class="rv-mono">{{ cmp.runB.originalFilename }} ({{ cmp.runB.sourceResolution ?? '—' }})</td>
                    <td>{{ cmp.runA.originalFilename === cmp.runB.originalFilename ? 'Igual' : 'Distinto' }}</td>
                  </tr>
                  <tr>
                    <td>Limiares / Amostragem</td>
                    <td class="rv-mono">conf={{ cmp.runA.confThreshold ?? '—' }} · iou={{ cmp.runA.iouThreshold ?? '—' }}{{ cmp.runA.sampleFps != null ? ' · ' + cmp.runA.sampleFps + 'fps' : '' }}</td>
                    <td class="rv-mono">conf={{ cmp.runB.confThreshold ?? '—' }} · iou={{ cmp.runB.iouThreshold ?? '—' }}{{ cmp.runB.sampleFps != null ? ' · ' + cmp.runB.sampleFps + 'fps' : '' }}</td>
                    <td>—</td>
                  </tr>
                  <tr>
                    <td>Dispositivo / Runtime</td>
                    <td class="rv-mono">{{ cmp.runA.deviceType }} ({{ cmp.runA.runtimeId ?? 'n/a' }})</td>
                    <td class="rv-mono">{{ cmp.runB.deviceType }} ({{ cmp.runB.runtimeId ?? 'n/a' }})</td>
                    <td>—</td>
                  </tr>
                  <tr>
                    <td>Detecções totais / Tracks</td>
                    <td>{{ cmp.runA.totalDetections }}{{ cmp.runA.uniqueTrackCount != null ? ' (' + cmp.runA.uniqueTrackCount + ' tracks)' : '' }}</td>
                    <td>{{ cmp.runB.totalDetections }}{{ cmp.runB.uniqueTrackCount != null ? ' (' + cmp.runB.uniqueTrackCount + ' tracks)' : '' }}</td>
                    <td class="rv-mono">{{ cmp.deltaDetections >= 0 ? '+' : '' }}{{ cmp.deltaDetections }}</td>
                  </tr>
                  <tr>
                    <td>FPS de processamento</td>
                    <td>{{ cmp.runA.processingFps != null ? (cmp.runA.processingFps | number: '1.1-2') : '—' }}</td>
                    <td>{{ cmp.runB.processingFps != null ? (cmp.runB.processingFps | number: '1.1-2') : '—' }}</td>
                    <td class="rv-mono">{{ cmp.deltaProcessingFps != null ? (cmp.deltaProcessingFps | number: '1.1-2') : '—' }}</td>
                  </tr>
                  <tr>
                    <td>Latência média / frame (ms)</td>
                    <td>{{ cmp.runA.avgFrameMs != null ? (cmp.runA.avgFrameMs | number: '1.1-2') : '—' }}</td>
                    <td>{{ cmp.runB.avgFrameMs != null ? (cmp.runB.avgFrameMs | number: '1.1-2') : '—' }}</td>
                    <td class="rv-mono">{{ cmp.deltaAvgFrameMs != null ? (cmp.deltaAvgFrameMs | number: '1.1-2') : '—' }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        }
      }
    </section>
  `,
})
export class LabComparePanelComponent {
  readonly runs = input<RunListItem[]>([]);
  readonly comparison = input<RunComparison | null>(null);
  readonly comparing = input<boolean>(false);

  readonly runA = signal<string | null>(null);
  readonly runB = signal<string | null>(null);

  readonly completedRuns = computed(() => this.runs().filter((r) => r.status === 'COMPLETED'));
  readonly compareRequested = output<{ runA: string; runB: string }>();
}

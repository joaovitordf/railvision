import { DatePipe, DecimalPipe } from '@angular/common';
import { Component, input } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { InspectionDetail, ProcessingJob, WagonResult } from '../../../core/models/inspection.models';
import { FormatBytesPipe } from '../../../shared/pipes/format-bytes.pipe';
import { FormatDurationPipe } from '../../../shared/pipes/format-duration.pipe';

@Component({
  selector: 'rv-inspection-summary-tab',
  standalone: true,
  imports: [DatePipe, DecimalPipe, MatIconModule, FormatBytesPipe, FormatDurationPipe],
  template: `
    <div class="rv-two-col-grid">
      <section class="rv-card">
        <div class="rv-card__header">
          <div>
            <h3 class="rv-card__title">Dados do vídeo e execução</h3>
            <p class="rv-card__subtitle">Metadados técnicos extraídos do arquivo original.</p>
          </div>
        </div>

        <dl class="rv-kv-list">
          <div class="rv-kv-row">
            <dt>Arquivo original</dt>
            <dd class="rv-mono">{{ detail().originalFilename }}</dd>
          </div>
          <div class="rv-kv-row">
            <dt>Tamanho / Formato</dt>
            <dd>{{ detail().fileSizeBytes | formatBytes }} · <span class="rv-mono">{{ detail().contentType }}</span></dd>
          </div>
          <div class="rv-kv-row">
            <dt>Resolução / Taxa</dt>
            <dd class="rv-mono">
              @if (detail().widthPx && detail().heightPx) {
                {{ detail().widthPx }}×{{ detail().heightPx }}
                @if (detail().fps) { · {{ detail().fps | number: '1.0-2' }} fps }
                @if (detail().codec) { · {{ detail().codec }} }
              } @else {
                —
              }
            </dd>
          </div>
          <div class="rv-kv-row">
            <dt>Duração</dt>
            <dd class="rv-mono">{{ detail().durationMs | formatDuration }}</dd>
          </div>
          <div class="rv-kv-row">
            <dt>Motor utilizado</dt>
            <dd class="rv-mono">{{ latestJob()?.engineType ?? 'Aguardando execução' }}</dd>
          </div>
          <div class="rv-kv-row">
            <dt>Criada em</dt>
            <dd>{{ detail().createdAt | date: 'dd/MM/yyyy HH:mm:ss' }}</dd>
          </div>
        </dl>

        @if (detail().notes) {
          <p style="margin: 14px 0 0; font-size: 0.85rem; color: var(--rv-text-secondary);">
            <strong>Observações:</strong> {{ detail().notes }}
          </p>
        }
      </section>

      <section class="rv-card">
        <div class="rv-card__header">
          <div>
            <h3 class="rv-card__title">Resumo da consolidação</h3>
            <p class="rv-card__subtitle">Contagem consolidada da última execução concluída.</p>
          </div>
        </div>

        <div class="rv-kpi-grid" style="grid-template-columns: repeat(2, minmax(130px, 1fr));">
          <div class="rv-kpi-card">
            <div class="rv-kpi-card__body">
              <span class="rv-kpi-card__label">Vagões registrados</span>
              <span class="rv-kpi-card__value">{{ wagons().length }}</span>
            </div>
            <div class="rv-kpi-card__icon" style="background: var(--rv-info-bg); color: var(--rv-info-text);">
              <mat-icon>view_carousel</mat-icon>
            </div>
          </div>

          <div class="rv-kpi-card">
            <div class="rv-kpi-card__body">
              <span class="rv-kpi-card__label">Revisão pendente</span>
              <span class="rv-kpi-card__value">{{ reviewCount() }}</span>
            </div>
            <div class="rv-kpi-card__icon" style="background: var(--rv-warn-bg); color: var(--rv-warn-text);">
              <mat-icon>flag</mat-icon>
            </div>
          </div>
        </div>

        <div class="rv-banner rv-banner--info" style="margin-top: 16px;">
          <mat-icon aria-hidden="true">info</mat-icon>
          <div>
            <strong>Escopo real desta etapa:</strong> Detecção (<span class="rv-mono">open_top_freight_wagon</span>) e
            tracking (<span class="rv-mono">ByteTrack</span>) no vídeo original. Classificação de material e nível ordinal
            permanecem <span class="rv-mono">not_evaluated</span> até o treinamento dos modelos dedicados.
          </div>
        </div>
      </section>
    </div>
  `,
})
export class InspectionSummaryTabComponent {
  readonly detail = input.required<InspectionDetail>();
  readonly latestJob = input<ProcessingJob | null>(null);
  readonly wagons = input<WagonResult[]>([]);
  readonly reviewCount = input<number>(0);
}

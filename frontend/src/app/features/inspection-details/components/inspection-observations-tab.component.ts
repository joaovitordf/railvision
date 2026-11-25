import { DecimalPipe } from '@angular/common';
import { Component, input, output } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { WagonObservation, WagonResult } from '../../../core/models/inspection.models';
import { EmptyStateComponent } from '../../../shared/components/empty-state/empty-state.component';

@Component({
  selector: 'rv-inspection-observations-tab',
  standalone: true,
  imports: [DecimalPipe, MatIconModule, EmptyStateComponent],
  template: `
    <section class="rv-card rv-card--table">
      <div class="rv-card__header">
        <div>
          <h3 class="rv-card__title">
            Observações por frame
            @if (selectedWagon(); as wagon) {
              · Vagão #{{ wagon.wagonIdentity.sequenceIndex }} (<span class="rv-mono">{{ wagon.wagonIdentity.trackId }}</span>)
            }
          </h3>
          <p class="rv-card__subtitle">
            Clique em qualquer linha para abrir o frame exato no player de vídeo.
          </p>
        </div>
      </div>

      @if (observations().length === 0) {
        <rv-empty-state
          icon="visibility_off"
          title="Nenhuma observação disponível"
          description="Selecione um vagão concluído para consultar as detecções por timestamp."
        />
      } @else {
        <div class="rv-table-wrap">
          <table class="rv-table">
            <thead>
              <tr>
                <th>Frame</th>
                <th>Timestamp</th>
                <th>Classe</th>
                <th>Confiança</th>
                <th>Bounding box (normalizado)</th>
                <th style="text-align: right;">Ação</th>
              </tr>
            </thead>
            <tbody>
              @for (obs of observations(); track obs.id) {
                <tr class="rv-row--selectable" (click)="openInVideo.emit(obs.frameTimestampMs)">
                  <td class="rv-mono">#{{ obs.frameIndex }}</td>
                  <td class="rv-mono">{{ obs.frameTimestampMs }} ms</td>
                  <td class="rv-mono">{{ obs.detectedClass }}</td>
                  <td>{{ obs.detectionConfidence | number: '1.2-2' }}</td>
                  <td class="rv-mono">
                    [{{ obs.boundingBox.x | number: '1.2-2' }}, {{ obs.boundingBox.y | number: '1.2-2' }},
                    {{ obs.boundingBox.width | number: '1.2-2' }}, {{ obs.boundingBox.height | number: '1.2-2' }}]
                  </td>
                  <td style="text-align: right;">
                    <button
                      type="button"
                      class="rv-btn rv-btn--secondary rv-btn--sm"
                      (click)="$event.stopPropagation(); openInVideo.emit(obs.frameTimestampMs)"
                    >
                      <mat-icon>play_circle</mat-icon>
                      <span>Ver no vídeo</span>
                    </button>
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
export class InspectionObservationsTabComponent {
  readonly selectedWagon = input<WagonResult | null>(null);
  readonly observations = input<WagonObservation[]>([]);
  readonly openInVideo = output<number>();
}

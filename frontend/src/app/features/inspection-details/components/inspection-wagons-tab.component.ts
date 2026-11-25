import { DecimalPipe } from '@angular/common';
import { Component, computed, input, output } from '@angular/core';
import { WagonResult } from '../../../core/models/inspection.models';
import { EmptyStateComponent } from '../../../shared/components/empty-state/empty-state.component';
import { LoadingLevelLabelPipe } from '../../../shared/pipes/loading-level-label.pipe';
import { MaterialLabelPipe } from '../../../shared/pipes/material-label.pipe';

@Component({
  selector: 'rv-inspection-wagons-tab',
  standalone: true,
  imports: [DecimalPipe, EmptyStateComponent, MaterialLabelPipe, LoadingLevelLabelPipe],
  template: `
    <div class="rv-page">
      <section class="rv-card rv-card--table">
        <div class="rv-card__header">
          <div>
            <h3 class="rv-card__title">Vagões consolidados</h3>
            <p class="rv-card__subtitle">
              Selecione um vagão para consultar motivos de revisão, flags de qualidade e suas observações por frame.
            </p>
          </div>
        </div>

        @if (wagons().length === 0) {
          <rv-empty-state
            icon="view_carousel"
            [title]="inspectionStatus() === 'COMPLETED' ? 'Nenhum vagão aberto detectado' : 'Sem resultados de vagões'"
            description="Os vagões consolidados aparecem após a conclusão do processamento."
          />
        } @else {
          <div class="rv-table-wrap">
            <table class="rv-table">
              <thead>
                <tr>
                  <th>Seq.</th>
                  <th>Track ID</th>
                  <th>Observações</th>
                  @if (hasEvaluatedAttributes()) {
                    <th>Material</th>
                    <th>Carregamento</th>
                  }
                  <th>Revisão</th>
                  <th>Origem</th>
                </tr>
              </thead>
              <tbody>
                @for (wagon of wagons(); track wagon.id) {
                  <tr
                    class="rv-row--selectable"
                    [class.rv-row--selected]="selectedWagon()?.id === wagon.id"
                    (click)="wagonSelected.emit(wagon)"
                  >
                    <td><strong>#{{ wagon.wagonIdentity.sequenceIndex }}</strong></td>
                    <td class="rv-mono">{{ wagon.wagonIdentity.trackId }}</td>
                    <td>{{ wagon.consolidation.observationsCount }}</td>
                    @if (hasEvaluatedAttributes()) {
                      <td>
                        @if (isEvaluatedMaterial(wagon)) {
                          {{ (wagon.material ? wagon.material.materialClass : null) | materialLabel: wagon.loadingEstimate.level }}
                          @if (wagon.material?.confidence != null) {
                            <span style="font-size: 0.76rem; color: var(--rv-text-muted);">
                              ({{ wagon.material!.confidence! | number: '1.2-2' }})
                            </span>
                          }
                        } @else {
                          <span class="rv-badge rv-badge--neutral">Não avaliado</span>
                        }
                      </td>
                      <td>
                        @if (wagon.loadingEstimate.level !== 'not_evaluated') {
                          {{ wagon.loadingEstimate.level | loadingLevelLabel }}
                        } @else {
                          <span class="rv-badge rv-badge--neutral">Não avaliado</span>
                        }
                      </td>
                    }
                    <td>
                      @if (wagon.reviewRequired) {
                        <span class="rv-badge rv-badge--warning" data-testid="review-required-badge">
                          <span class="rv-badge__dot" aria-hidden="true"></span>
                          <span>Revisão necessária</span>
                        </span>
                      } @else {
                        <span class="rv-badge rv-badge--success">
                          <span class="rv-badge__dot" aria-hidden="true"></span>
                          <span>OK</span>
                        </span>
                      }
                    </td>
                    <td>
                      @if (wagon.isExample) {
                        <span class="rv-badge rv-badge--warning">Exemplo fictício</span>
                      } @else {
                        <span class="rv-badge rv-badge--info">Inferência real</span>
                      }
                    </td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
        }
      </section>

      @if (selectedWagon(); as wagon) {
        <section class="rv-card">
          <div class="rv-card__header">
            <div>
              <h3 class="rv-card__title">
                Vagão #{{ wagon.wagonIdentity.sequenceIndex }} (<span class="rv-mono">{{ wagon.wagonIdentity.trackId }}</span>)
              </h3>
              <p class="rv-card__subtitle">
                ID físico (OCR): <span class="rv-mono">{{ wagon.wagonIdentity.physicalId ?? 'não avaliado no MVP' }}</span>
              </p>
            </div>
          </div>

          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 18px;">
            <div>
              <strong style="font-size: 0.85rem;">Motivos de revisão</strong>
              @if (wagon.reviewReasons.length === 0) {
                <div style="color: var(--rv-text-muted); font-size: 0.84rem; margin-top: 6px;">Nenhum sinalizado.</div>
              } @else {
                <ul style="margin: 6px 0 0; padding-left: 18px; font-size: 0.84rem;">
                  @for (reason of wagon.reviewReasons; track reason) {
                    <li class="rv-mono">{{ reason }}</li>
                  }
                </ul>
              }
            </div>

            <div>
              <strong style="font-size: 0.85rem;">Qualidade e consolidação</strong>
              <div style="margin-top: 6px; font-size: 0.84rem; display: flex; flex-direction: column; gap: 8px;">
                <div>
                  Método: <span class="rv-mono">{{ wagon.consolidation.method ?? 'single_stage' }}</span>
                </div>
                @if (wagon.loadingEstimate.level === 'not_evaluated') {
                  <div>
                    Material / Carregamento: <span class="rv-badge rv-badge--neutral">Não avaliado</span>
                  </div>
                }
                <div style="display: flex; flex-wrap: wrap; gap: 6px;">
                  @if (wagon.qualityFlags.length > 0) {
                    @for (flag of wagon.qualityFlags; track flag) {
                      <span class="rv-badge rv-badge--warning rv-mono">{{ flag }}</span>
                    }
                  } @else {
                    <span style="color: var(--rv-text-muted);">Sem flags de qualidade.</span>
                  }
                </div>
              </div>
            </div>

            <div>
              <strong style="font-size: 0.85rem;">Modelos associados</strong>
              <div style="margin-top: 6px; font-size: 0.82rem; display: flex; flex-direction: column; gap: 4px;">
                @for (entry of modelVersionEntries(wagon); track entry[0]) {
                  <div>
                    <span class="rv-mono" style="color: var(--rv-text-muted);">{{ entry[0] }}</span>:
                    <span class="rv-mono">{{ entry[1] }}</span>
                  </div>
                }
              </div>
            </div>
          </div>
        </section>
      }
    </div>
  `,
})
export class InspectionWagonsTabComponent {
  readonly inspectionStatus = input.required<string>();
  readonly wagons = input<WagonResult[]>([]);
  readonly selectedWagon = input<WagonResult | null>(null);
  readonly wagonSelected = output<WagonResult>();

  protected readonly hasEvaluatedAttributes = computed(() =>
    this.wagons().some((w) => w.loadingEstimate.level !== 'not_evaluated' || this.isEvaluatedMaterial(w))
  );

  protected isEvaluatedMaterial(wagon: WagonResult): boolean {
    if (wagon.loadingEstimate.level === 'empty') {
      return true;
    }
    return Boolean(wagon.material?.materialClass && wagon.material.materialClass !== 'not_evaluated');
  }

  protected modelVersionEntries(wagon: WagonResult): [string, string][] {
    return Object.entries(wagon.modelVersions ?? {});
  }
}

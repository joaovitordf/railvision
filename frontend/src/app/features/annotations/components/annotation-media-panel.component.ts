import { Component, computed, inject, input, output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { ModelCheckpointSummary } from '../../../core/models/runtime.models';
import { extractApiErrorMessage } from '../../../core/services/api-error.util';
import { NotificationService } from '../../../core/services/notification.service';
import { EmptyStateComponent } from '../../../shared/components/empty-state/empty-state.component';
import { AnnotationItemSummary, AnnotationProjectDetail } from '../models/annotations.models';
import { AnnotationsApiService } from '../services/annotations-api.service';
import { AnnotationVideoExtractorComponent } from './annotation-video-extractor.component';

const STATUS_LABELS: Record<string, string> = {
  pending: 'Pendente',
  pre_annotated: 'Pré-anotada',
  in_review: 'Em revisão',
  reviewed: 'Revisada',
  approved: 'Aprovada',
  rejected: 'Rejeitada',
};

@Component({
  selector: 'rv-annotation-media-panel',
  standalone: true,
  imports: [
    FormsModule,
    MatIconModule,
    MatProgressBarModule,
    EmptyStateComponent,
    AnnotationVideoExtractorComponent,
  ],
  template: `
    <div class="rv-page">
      <section class="rv-card">
        <div class="rv-card__header">
          <div>
            <h3 class="rv-card__title">Importação e fluxo assistido</h3>
            <p class="rv-card__subtitle">
              Importe imagens, execute pré-anotações com IA e exporte pacotes YOLO 1.1 após revisão humana.
            </p>
          </div>

          <div class="rv-header-actions">
            <label class="rv-btn rv-btn--secondary" [style.opacity]="uploadingImages() ? '0.6' : '1'">
              <mat-icon>add_photo_alternate</mat-icon>
              <span>{{ uploadingImages() ? 'Enviando...' : 'Enviar imagens' }}</span>
              <input
                type="file"
                multiple
                accept=".jpg,.jpeg,.png,.bmp,.webp,image/*"
                [disabled]="uploadingImages()"
                (change)="onImagesSelected($event)"
                style="display: none;"
              />
            </label>

            <button
              type="button"
              class="rv-btn rv-btn--secondary"
              [disabled]="busyAction()"
              (click)="importCuration()"
            >
              <mat-icon>inbox</mat-icon>
              <span>Importar da curadoria</span>
            </button>

            <label class="rv-btn rv-btn--secondary" [style.opacity]="busyAction() ? '0.6' : '1'">
              <mat-icon>archive</mat-icon>
              <span>Importar CVAT YOLO (.zip)</span>
              <input
                type="file"
                accept=".zip,application/zip"
                [disabled]="busyAction()"
                (change)="onImportYoloZip($event)"
                style="display: none;"
              />
            </label>

            <button
              type="button"
              class="rv-btn rv-btn--primary"
              data-testid="export-yolo-btn"
              [disabled]="busyAction() || project().reviewedCount === 0"
              (click)="exportYolo()"
            >
              <mat-icon>download</mat-icon>
              <span>Exportar YOLO 1.1 ({{ project().reviewedCount }})</span>
            </button>
          </div>
        </div>

        <div class="rv-form-grid">
          @if (checkpoints().length > 0) {
            <div class="rv-field">
              <label class="rv-label" for="ann-pre-model">Modelo de pré-anotação IA</label>
              <select
                id="ann-pre-model"
                class="rv-select"
                [ngModel]="selectedModelId()"
                (ngModelChange)="selectedModelId.set($event)"
              >
                @for (cp of checkpoints(); track cp.id) {
                  <option [ngValue]="cp.id">{{ cp.id }} [{{ cp.status }}]</option>
                }
              </select>
            </div>
          }

          <div class="rv-field" style="max-width: 180px;">
            <label class="rv-label" for="ann-pre-conf">Confiança mínima</label>
            <input
              id="ann-pre-conf"
              class="rv-input"
              type="number"
              min="0.05"
              max="0.95"
              step="0.05"
              [ngModel]="preConf()"
              (ngModelChange)="preConf.set(+$event)"
            />
          </div>

          <div style="display: flex;">
            <button
              type="button"
              class="rv-btn rv-btn--secondary"
              data-testid="preannotate-batch-btn"
              [disabled]="busyAction() || project().itemCount === 0"
              (click)="runBatchPreannotate()"
            >
              <mat-icon>auto_awesome</mat-icon>
              <span>{{ preannotating() ? 'Pré-anotando...' : 'Pré-anotar pendentes' }}</span>
            </button>
          </div>
        </div>

        @if (busyAction()) {
          <mat-progress-bar mode="indeterminate" style="margin-top: 12px;" />
        }
      </section>

      <rv-annotation-video-extractor
        [projectId]="project().id"
        [videos]="project().videos"
        (framesImported)="projectUpdated.emit()"
      />

      <section class="rv-card">
        <div class="rv-card__header">
          <div>
            <h3 class="rv-card__title">Imagens e frames do projeto ({{ filteredItems().length }})</h3>
            <p class="rv-card__subtitle">
              Origens não são promovidas automaticamente para os splits oficiais; aprove cada item no editor.
            </p>
          </div>

          <div class="rv-filter-bar">
            <div class="rv-search">
              <mat-icon aria-hidden="true">search</mat-icon>
              <input
                class="rv-input"
                [ngModel]="search()"
                (ngModelChange)="search.set($event)"
                placeholder="Filtrar arquivo ou vídeo..."
              />
            </div>

            <select
              class="rv-select"
              style="width: 185px;"
              aria-label="Filtrar por estado"
              [ngModel]="statusFilter()"
              (ngModelChange)="statusFilter.set($event)"
            >
              <option value="ALL">Todos os estados</option>
              <option value="pending">Pendente</option>
              <option value="pre_annotated">Pré-anotada</option>
              <option value="in_review">Em revisão</option>
              <option value="reviewed">Revisada</option>
              <option value="approved">Aprovada</option>
              <option value="rejected">Rejeitada</option>
            </select>
          </div>
        </div>

        @if (filteredItems().length === 0) {
          <rv-empty-state
            icon="image_search"
            title="Nenhum item corresponde ao filtro"
            description="Envie imagens acima, extraia frames de vídeo ou ajuste os filtros."
          />
        } @else {
          <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(225px, 1fr)); gap: 16px;">
            @for (item of filteredItems(); track item.id) {
              <article
                style="border: 1px solid var(--rv-border); border-radius: var(--rv-radius-md); overflow: hidden; display: flex; flex-direction: column; background: var(--rv-bg-surface); box-shadow: var(--rv-shadow-xs);"
              >
                <div style="position: relative; cursor: pointer;" (click)="openInEditor.emit(item.id)">
                  <img
                    [src]="item.thumbnailUrl"
                    [alt]="item.filename"
                    style="width: 100%; height: 142px; object-fit: cover; display: block; background: #0b111c;"
                  />
                  <span
                    class="rv-badge"
                    [class.rv-badge--success]="item.status === 'approved' || item.status === 'reviewed'"
                    [class.rv-badge--warning]="item.status === 'pre_annotated' || item.status === 'in_review'"
                    [class.rv-badge--error]="item.status === 'rejected'"
                    [class.rv-badge--neutral]="item.status === 'pending'"
                    style="position: absolute; top: 8px; left: 8px;"
                  >
                    <span class="rv-badge__dot" aria-hidden="true"></span>
                    {{ statusLabel(item.status) }}
                  </span>
                  @if (item.reservedBenchmarkWarning) {
                    <span class="rv-badge rv-badge--error" style="position: absolute; top: 8px; right: 8px;">
                      Reservado
                    </span>
                  }
                </div>

                <div style="padding: 12px 14px; display: flex; flex-direction: column; gap: 6px; flex: 1;">
                  <strong style="font-size: 0.85rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    {{ item.filename }}
                  </strong>
                  <div style="font-size: 0.77rem; color: var(--rv-text-muted); display: flex; justify-content: space-between;">
                    <span>{{ item.boxCount }} caixas</span>
                    <span class="rv-mono">{{ item.widthPx }}×{{ item.heightPx }}</span>
                  </div>
                  @if (item.sourceVideoFilename) {
                    <div class="rv-mono" style="font-size: 0.73rem; color: var(--rv-text-muted);">
                      Frame #{{ item.frameIndex }} · {{ item.sourceOrigin }}
                    </div>
                  }
                  <div style="margin-top: auto; padding-top: 8px; display: flex; align-items: center; gap: 8px;">
                    <button
                      type="button"
                      class="rv-btn rv-btn--primary rv-btn--sm"
                      (click)="openInEditor.emit(item.id)"
                      style="flex: 1;"
                    >
                      <mat-icon>edit</mat-icon>
                      <span>Revisar</span>
                    </button>
                    <button
                      type="button"
                      class="rv-icon-btn"
                      style="width: 32px; height: 32px;"
                      (click)="deleteItem(item)"
                      aria-label="Remover item"
                      title="Remover item"
                    >
                      <mat-icon>delete_outline</mat-icon>
                    </button>
                  </div>
                </div>
              </article>
            }
          </div>
        }
      </section>
    </div>
  `,
})
export class AnnotationMediaPanelComponent {
  private readonly api = inject(AnnotationsApiService);
  private readonly notifications = inject(NotificationService);

  readonly project = input.required<AnnotationProjectDetail>();
  readonly checkpoints = input<ModelCheckpointSummary[]>([]);
  readonly projectUpdated = output<void>();
  readonly openInEditor = output<string>();

  protected readonly uploadingImages = signal<boolean>(false);
  protected readonly preannotating = signal<boolean>(false);
  protected readonly busyAction = signal<boolean>(false);
  protected readonly selectedModelId = signal<string>('exp04c_yolo26n_opentop_640');
  protected readonly preConf = signal<number>(0.25);
  protected readonly search = signal<string>('');
  protected readonly statusFilter = signal<string>('ALL');

  protected readonly filteredItems = computed(() => {
    const q = this.search().trim().toLowerCase();
    const st = this.statusFilter();
    return this.project().items.filter((item) => {
      if (st !== 'ALL' && item.status !== st) return false;
      if (q) {
        const hay = `${item.filename} ${item.sourceOrigin} ${item.sourceVideoFilename ?? ''}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  });

  protected statusLabel(st: string): string {
    return STATUS_LABELS[st] ?? st;
  }

  protected onImagesSelected(event: Event): void {
    const inputEl = event.target as HTMLInputElement;
    const files = inputEl.files ? Array.from(inputEl.files) : [];
    inputEl.value = '';
    if (files.length === 0) return;

    this.uploadingImages.set(true);
    this.api.uploadImages(this.project().id, files).subscribe({
      next: (created) => {
        this.uploadingImages.set(false);
        this.notifications.success(`${created.length} imagem(ns) importada(s).`);
        this.projectUpdated.emit();
      },
      error: (err) => {
        this.uploadingImages.set(false);
        this.notifications.error(extractApiErrorMessage(err, 'Falha ao enviar imagens.'));
      },
    });
  }

  protected runBatchPreannotate(): void {
    this.preannotating.set(true);
    this.busyAction.set(true);
    this.api
      .preannotateBatch(this.project().id, {
        modelId: this.selectedModelId(),
        conf: this.preConf(),
        iou: 0.45,
      })
      .subscribe({
        next: (res) => {
          this.preannotating.set(false);
          this.busyAction.set(false);
          this.notifications.success(
            `Pré-anotação concluída: ${res.totalSuggestions} sugestões em ${res.processedItems} item(ns).`
          );
          this.projectUpdated.emit();
        },
        error: (err) => {
          this.preannotating.set(false);
          this.busyAction.set(false);
          this.notifications.error(extractApiErrorMessage(err, 'Falha na pré-anotação em lote.'));
        },
      });
  }

  protected importCuration(): void {
    this.busyAction.set(true);
    this.api.importCurationInbox(this.project().id).subscribe({
      next: (res) => {
        this.busyAction.set(false);
        this.notifications.info(
          `Curadoria: ${res.importedCount} importado(s), ${res.skippedDuplicates} duplicado(s) ignorado(s).`
        );
        this.projectUpdated.emit();
      },
      error: (err) => {
        this.busyAction.set(false);
        this.notifications.error(extractApiErrorMessage(err, 'Falha ao importar curation-inbox.'));
      },
    });
  }

  protected exportYolo(): void {
    this.busyAction.set(true);
    this.api.exportYoloZip(this.project().id, false).subscribe({
      next: (blob) => {
        this.busyAction.set(false);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `railvision-${this.project().campaignId}-yolo11.zip`;
        a.click();
        URL.revokeObjectURL(url);
        this.notifications.success('Pacote YOLO 1.1 exportado com sucesso.');
      },
      error: (err) => {
        this.busyAction.set(false);
        this.notifications.error(extractApiErrorMessage(err, 'Falha ao exportar pacote YOLO.'));
      },
    });
  }

  protected onImportYoloZip(event: Event): void {
    const inputEl = event.target as HTMLInputElement;
    const file = inputEl.files?.[0];
    inputEl.value = '';
    if (!file) return;
    this.busyAction.set(true);
    this.api.importYoloZip(this.project().id, file).subscribe({
      next: (items) => {
        this.busyAction.set(false);
        this.notifications.success(`${items.length} item(ns) importado(s) do pacote YOLO.`);
        this.projectUpdated.emit();
      },
      error: (err) => {
        this.busyAction.set(false);
        this.notifications.error(extractApiErrorMessage(err, 'Falha ao importar pacote YOLO (.zip).'));
      },
    });
  }

  protected deleteItem(item: AnnotationItemSummary): void {
    this.api.deleteItem(this.project().id, item.id).subscribe({
      next: () => {
        this.notifications.info(`Item '${item.filename}' removido.`);
        this.projectUpdated.emit();
      },
      error: (err) => {
        this.notifications.error(extractApiErrorMessage(err, 'Falha ao excluir item.'));
      },
    });
  }
}

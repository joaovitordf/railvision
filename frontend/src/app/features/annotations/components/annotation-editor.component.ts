import { DatePipe, DecimalPipe } from '@angular/common';
import {
  Component,
  HostListener,
  ViewChild,
  computed,
  effect,
  inject,
  input,
  output,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { ModelCheckpointSummary } from '../../../core/models/runtime.models';
import { extractApiErrorMessage } from '../../../core/services/api-error.util';
import { NotificationService } from '../../../core/services/notification.service';
import {
  AnnotationBox,
  AnnotationItemDetail,
  AnnotationItemStatus,
  AnnotationItemSummary,
} from '../models/annotations.models';
import { AnnotationsApiService } from '../services/annotations-api.service';
import { AnnotationCanvasComponent, EditorTool } from './annotation-canvas.component';

@Component({
  selector: 'rv-annotation-editor',
  standalone: true,
  imports: [
    DatePipe,
    DecimalPipe,
    FormsModule,
    MatIconModule,
    MatTooltipModule,
    AnnotationCanvasComponent,
  ],
  template: `
    <div class="rv-editor-shell" data-testid="annotation-editor-shell">
      <header class="rv-editor-topbar">
        <div class="rv-editor-topbar__left">
          <button type="button" class="rv-studio-btn" (click)="confirmExit()">
            <mat-icon>arrow_back</mat-icon>
            <span>Dataset</span>
          </button>

          <div class="rv-studio-pager">
            <button
              type="button"
              class="rv-studio-icon-btn"
              [disabled]="!hasPrev()"
              (click)="navigateBy(-1)"
              matTooltip="Imagem anterior (←)"
              aria-label="Imagem anterior"
            >
              <mat-icon>chevron_left</mat-icon>
            </button>
            <span class="rv-mono rv-studio-pager__label">
              {{ currentIndex() + 1 }} / {{ items().length }}
            </span>
            <button
              type="button"
              class="rv-studio-icon-btn"
              [disabled]="!hasNext()"
              (click)="navigateBy(1)"
              matTooltip="Próxima imagem (→)"
              aria-label="Próxima imagem"
            >
              <mat-icon>chevron_right</mat-icon>
            </button>
          </div>

          @if (detail(); as det) {
            <span class="rv-mono rv-editor-filename" [title]="det.filename">
              {{ det.filename }} ({{ det.widthPx }}×{{ det.heightPx }})
              @if (det.frameIndex != null) { · frame #{{ det.frameIndex }} }
            </span>
            @if (dirty()) {
              <span class="rv-studio-pill rv-studio-pill--warn">Não salvo</span>
            }
          }
        </div>

        <div class="rv-editor-topbar__right">
          <button
            type="button"
            class="rv-studio-btn rv-studio-btn--amber"
            data-testid="editor-preannotate-btn"
            [disabled]="saving() || preannotating()"
            (click)="runSinglePreannotate()"
          >
            <mat-icon>auto_awesome</mat-icon>
            <span>{{ preannotating() ? 'Sugerindo...' : 'Pré-anotar IA' }}</span>
          </button>

          <button
            type="button"
            class="rv-studio-btn"
            [class.rv-studio-btn--amber-active]="noObjectsConfirmed()"
            data-testid="editor-no-objects-btn"
            (click)="toggleNoObjects()"
          >
            <mat-icon>block</mat-icon>
            <span>{{ noObjectsConfirmed() ? 'Sem vagões confirmado' : 'Marcar negativo (N)' }}</span>
          </button>

          <button
            type="button"
            class="rv-studio-btn rv-studio-btn--blue"
            data-testid="editor-save-btn"
            [disabled]="saving()"
            (click)="saveCurrent()"
          >
            <mat-icon>save</mat-icon>
            <span>Salvar</span>
          </button>

          <button
            type="button"
            class="rv-studio-btn rv-studio-btn--primary"
            data-testid="editor-approve-btn"
            [disabled]="saving()"
            (click)="saveCurrent('approved')"
          >
            <mat-icon>check_circle</mat-icon>
            <span>Aprovar revisão</span>
          </button>
        </div>
      </header>

      <div class="rv-editor-workspace">
        <aside class="rv-tool-rail" aria-label="Ferramentas do editor">
          <button
            type="button"
            class="rv-tool-btn"
            [class.rv-tool-btn--active]="tool() === 'select'"
            (click)="tool.set('select')"
            matTooltip="Selecionar / Mover (V)"
            matTooltipPosition="right"
            aria-label="Selecionar ou mover"
          >
            <mat-icon>near_me</mat-icon>
          </button>
          <button
            type="button"
            class="rv-tool-btn"
            data-testid="tool-draw-btn"
            [class.rv-tool-btn--active]="tool() === 'draw'"
            (click)="tool.set('draw')"
            matTooltip="Desenhar caixa (R)"
            matTooltipPosition="right"
            aria-label="Desenhar bounding box"
          >
            <mat-icon>crop_square</mat-icon>
          </button>
          <button
            type="button"
            class="rv-tool-btn"
            [class.rv-tool-btn--active]="tool() === 'pan'"
            (click)="tool.set('pan')"
            matTooltip="Mover tela (H)"
            matTooltipPosition="right"
            aria-label="Mover tela"
          >
            <mat-icon>pan_tool</mat-icon>
          </button>

          <div class="rv-tool-divider"></div>

          <button type="button" class="rv-tool-btn" (click)="canvasRef?.zoomIn()" matTooltip="Zoom +" matTooltipPosition="right">
            <mat-icon>zoom_in</mat-icon>
          </button>
          <button type="button" class="rv-tool-btn" (click)="canvasRef?.zoomOut()" matTooltip="Zoom -" matTooltipPosition="right">
            <mat-icon>zoom_out</mat-icon>
          </button>
          <button type="button" class="rv-tool-btn" (click)="canvasRef?.fitScreen()" matTooltip="Ajustar tela (0)" matTooltipPosition="right">
            <mat-icon>fullscreen</mat-icon>
          </button>

          <div class="rv-tool-divider"></div>

          <button type="button" class="rv-tool-btn" [disabled]="undoStack().length === 0" (click)="undo()" matTooltip="Desfazer (Ctrl+Z)" matTooltipPosition="right">
            <mat-icon>undo</mat-icon>
          </button>
          <button type="button" class="rv-tool-btn" [disabled]="redoStack().length === 0" (click)="redo()" matTooltip="Refazer (Ctrl+Y)" matTooltipPosition="right">
            <mat-icon>redo</mat-icon>
          </button>
          <button type="button" class="rv-tool-btn" [disabled]="!selectedBoxId()" (click)="deleteSelectedBox()" matTooltip="Excluir caixa (Del)" matTooltipPosition="right">
            <mat-icon>delete</mat-icon>
          </button>

          <div style="margin-top: auto; display: flex; flex-direction: column; gap: 4px;">
            <button
              type="button"
              class="rv-tool-btn"
              [class.rv-tool-btn--active]="inspectorCollapsed()"
              (click)="inspectorCollapsed.set(!inspectorCollapsed())"
              matTooltip="Mostrar / ocultar painel lateral"
              matTooltipPosition="right"
            >
              <mat-icon>{{ inspectorCollapsed() ? 'vertical_split' : 'chevron_right' }}</mat-icon>
            </button>
            <button
              type="button"
              class="rv-tool-btn"
              [class.rv-tool-btn--active]="showShortcuts()"
              (click)="showShortcuts.set(!showShortcuts())"
              matTooltip="Atalhos de teclado (?)"
              matTooltipPosition="right"
            >
              <mat-icon>keyboard</mat-icon>
            </button>
          </div>
        </aside>

        <div class="rv-editor-canvas-wrap">
          <rv-annotation-canvas
            #canvasRef
            [imageUrl]="detail()?.imageUrl ?? null"
            [boxes]="boxes()"
            [selectedBoxId]="selectedBoxId()"
            [tool]="tool()"
            [activeClassName]="activeClassName()"
            [noObjectsConfirmed]="noObjectsConfirmed()"
            (boxSelected)="selectedBoxId.set($event)"
            (boxCreated)="onBoxCreated($event)"
            (boxUpdated)="onBoxUpdated($event)"
          />

          @if (showShortcuts()) {
            <div class="rv-shortcuts-modal" role="dialog" aria-label="Atalhos do editor">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <strong style="font-size: 0.88rem; color: #f8fafc;">Atalhos de teclado</strong>
                <button type="button" class="rv-studio-icon-btn" (click)="showShortcuts.set(false)">
                  <mat-icon>close</mat-icon>
                </button>
              </div>
              <div style="display: grid; grid-template-columns: 92px 1fr; gap: 6px 12px; font-size: 0.8rem; color: #cbd5e1;">
                <span class="rv-mono" style="color: #93c5fd;">R</span><span>Desenhar caixa</span>
                <span class="rv-mono" style="color: #93c5fd;">V</span><span>Selecionar / Mover / Redimensionar</span>
                <span class="rv-mono" style="color: #93c5fd;">H / Shift</span><span>Arrastar / Pan na imagem</span>
                <span class="rv-mono" style="color: #93c5fd;">N</span><span>Alternar sem objetos (negativo)</span>
                <span class="rv-mono" style="color: #93c5fd;">Del</span><span>Excluir caixa selecionada</span>
                <span class="rv-mono" style="color: #93c5fd;">Ctrl+Z / Y</span><span>Desfazer / Refazer</span>
                <span class="rv-mono" style="color: #93c5fd;">Ctrl+S</span><span>Salvar revisão humana</span>
                <span class="rv-mono" style="color: #93c5fd;">← / →</span><span>Imagem anterior / próxima</span>
                <span class="rv-mono" style="color: #93c5fd;">0 / Scroll</span><span>Ajustar à tela / Zoom no cursor</span>
              </div>
            </div>
          }
        </div>

        @if (!inspectorCollapsed()) {
          <aside class="rv-editor-inspector" aria-label="Painel de objetos e revisão">
            @if (detail(); as det) {
              @if (det.reservedBenchmarkWarning) {
                <div class="rv-studio-alert">
                  <mat-icon>warning</mat-icon>
                  <div>Mídia reservada (<span class="rv-mono">vid_01/09/16</span>). Não promover para treino.</div>
                </div>
              }

              <div class="rv-studio-field">
                <label class="rv-studio-label" for="studio-class-select">Classe ativa</label>
                <select
                  id="studio-class-select"
                  class="rv-studio-select"
                  [ngModel]="activeClassName()"
                  (ngModelChange)="activeClassName.set($event)"
                >
                  @for (cls of classes(); track cls) {
                    <option [ngValue]="cls">{{ cls }}</option>
                  }
                </select>
              </div>

              <div class="rv-studio-field">
                <label class="rv-studio-label" for="studio-status-select">Estado do item</label>
                <select
                  id="studio-status-select"
                  class="rv-studio-select"
                  [ngModel]="itemStatus()"
                  (ngModelChange)="onStatusChange($event)"
                >
                  <option value="pending">Pendente</option>
                  <option value="pre_annotated">Pré-anotada (IA)</option>
                  <option value="in_review">Em revisão</option>
                  <option value="reviewed">Revisada</option>
                  <option value="approved">Aprovada</option>
                  <option value="rejected">Rejeitada</option>
                </select>
              </div>

              <div class="rv-studio-section-head">
                <strong>Objetos ({{ boxes().length }})</strong>
                @if (aiSuggestionsCount() > 0) {
                  <button
                    type="button"
                    class="rv-studio-btn rv-studio-btn--xs"
                    (click)="acceptAllAiSuggestions()"
                  >
                    <mat-icon>done_all</mat-icon>
                    <span>Aceitar IA ({{ aiSuggestionsCount() }})</span>
                  </button>
                }
              </div>

              <div class="rv-studio-box-list">
                @if (boxes().length === 0) {
                  <div class="rv-studio-empty">
                    {{ noObjectsConfirmed() ? 'Confirmado sem objetos.' : 'Nenhuma caixa ainda. Pressione R para desenhar ou clique em Pré-anotar IA.' }}
                  </div>
                } @else {
                  @for (box of boxes(); track box.id; let idx = $index) {
                    <div
                      class="rv-studio-box-item"
                      [class.rv-studio-box-item--selected]="selectedBoxId() === box.id"
                      [class.rv-studio-box-item--ai]="box.origin === 'ai_suggestion'"
                      (click)="selectedBoxId.set(box.id)"
                    >
                      <div style="min-width: 0;">
                        <div class="rv-mono rv-studio-box-item__title">
                          #{{ idx + 1 }} {{ box.className }}
                        </div>
                        <div class="rv-studio-box-item__meta">
                          {{ box.origin === 'ai_suggestion' ? 'Sugestão IA' : box.origin === 'ai_edited' ? 'IA editada' : box.origin === 'ai_accepted' ? 'IA aceita' : 'Manual' }}
                          @if (box.confidence != null) {
                            · {{ (box.confidence * 100) | number: '1.0-0' }}%
                          }
                        </div>
                      </div>
                      <div style="display: flex; gap: 4px; flex-shrink: 0;">
                        @if (box.origin === 'ai_suggestion') {
                          <button
                            type="button"
                            class="rv-studio-mini-btn rv-studio-mini-btn--accept"
                            (click)="$event.stopPropagation(); acceptBox(box.id)"
                            matTooltip="Aceitar sugestão"
                            aria-label="Aceitar sugestão"
                          >
                            <mat-icon>check</mat-icon>
                          </button>
                        }
                        <button
                          type="button"
                          class="rv-studio-mini-btn rv-studio-mini-btn--delete"
                          (click)="$event.stopPropagation(); removeBoxById(box.id)"
                          matTooltip="Remover caixa"
                          aria-label="Remover caixa"
                        >
                          <mat-icon>close</mat-icon>
                        </button>
                      </div>
                    </div>
                  }
                }
              </div>

              <div class="rv-studio-field" style="margin-top: auto;">
                <label class="rv-studio-label" for="studio-notes">Notas de revisão</label>
                <textarea
                  id="studio-notes"
                  class="rv-studio-textarea"
                  rows="2"
                  placeholder="Observações sobre oclusão, sombra ou qualidade..."
                  [ngModel]="reviewNotes()"
                  (ngModelChange)="reviewNotes.set($event); dirty.set(true)"
                ></textarea>
              </div>

              <div class="rv-studio-meta">
                <div><span>Campanha</span><strong class="rv-mono">{{ det.campaignId }}</strong></div>
                <div><span>Origem</span><strong class="rv-mono">{{ det.sourceOrigin }}</strong></div>
                @if (det.preannotationModelId) {
                  <div><span>Modelo IA</span><strong class="rv-mono">{{ det.preannotationModelId }}</strong></div>
                }
                <div><span>FPs / FNs</span><strong class="rv-mono">{{ fpRemoved() }} / {{ fnAdded() }}</strong></div>
                @if (det.reviewedAt) {
                  <div><span>Revisado</span><strong>{{ det.reviewedAt | date: 'dd/MM HH:mm' }}</strong></div>
                }
              </div>
            }
          </aside>
        }
      </div>
    </div>
  `,
})
export class AnnotationEditorComponent {
  private readonly api = inject(AnnotationsApiService);
  private readonly notifications = inject(NotificationService);

  @ViewChild('canvasRef') protected canvasRef?: AnnotationCanvasComponent;

  readonly projectId = input.required<string>();
  readonly items = input<AnnotationItemSummary[]>([]);
  readonly itemId = input.required<string>();
  readonly classes = input<string[]>(['open_top_freight_wagon']);
  readonly checkpoints = input<ModelCheckpointSummary[]>([]);

  readonly itemSelected = output<string>();
  readonly itemSaved = output<void>();
  readonly exitEditor = output<void>();

  protected readonly detail = signal<AnnotationItemDetail | null>(null);
  protected readonly boxes = signal<AnnotationBox[]>([]);
  protected readonly selectedBoxId = signal<string | null>(null);
  protected readonly tool = signal<EditorTool>('select');
  protected readonly activeClassName = signal<string>('open_top_freight_wagon');
  protected readonly itemStatus = signal<AnnotationItemStatus>('in_review');
  protected readonly noObjectsConfirmed = signal<boolean>(false);
  protected readonly reviewNotes = signal<string>('');
  protected readonly fpRemoved = signal<number>(0);
  protected readonly fnAdded = signal<number>(0);
  protected readonly dirty = signal<boolean>(false);
  protected readonly saving = signal<boolean>(false);
  protected readonly preannotating = signal<boolean>(false);
  protected readonly inspectorCollapsed = signal<boolean>(false);
  protected readonly showShortcuts = signal<boolean>(false);
  protected readonly undoStack = signal<AnnotationBox[][]>([]);
  protected readonly redoStack = signal<AnnotationBox[][]>([]);

  protected readonly currentIndex = computed(() =>
    Math.max(0, this.items().findIndex((it) => it.id === this.itemId()))
  );
  protected readonly hasPrev = computed(() => this.currentIndex() > 0);
  protected readonly hasNext = computed(() => this.currentIndex() < this.items().length - 1);
  protected readonly aiSuggestionsCount = computed(
    () => this.boxes().filter((b) => b.origin === 'ai_suggestion').length
  );

  constructor() {
    effect(() => {
      const pid = this.projectId();
      const iid = this.itemId();
      if (pid && iid) {
        this.loadItem(pid, iid);
      }
    });
  }

  @HostListener('window:keydown', ['$event'])
  protected onKeyDown(event: KeyboardEvent): void {
    const tag = (event.target as HTMLElement | null)?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
      return;
    }

    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
      event.preventDefault();
      this.saveCurrent();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
      event.preventDefault();
      this.undo();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y') {
      event.preventDefault();
      this.redo();
      return;
    }

    switch (event.key.toLowerCase()) {
      case 'v':
        this.tool.set('select');
        break;
      case 'r':
        this.tool.set('draw');
        break;
      case 'h':
        this.tool.set('pan');
        break;
      case 'n':
        this.toggleNoObjects();
        break;
      case '0':
        this.canvasRef?.fitScreen();
        break;
      case '?':
        this.showShortcuts.update((v) => !v);
        break;
      case 'delete':
      case 'backspace':
        this.deleteSelectedBox();
        break;
      case 'arrowleft':
        if (this.hasPrev()) this.navigateBy(-1);
        break;
      case 'arrowright':
        if (this.hasNext()) this.navigateBy(1);
        break;
    }
  }

  private loadItem(projectId: string, itemId: string): void {
    this.api.getItem(projectId, itemId).subscribe({
      next: (det) => {
        this.detail.set(det);
        this.boxes.set(det.boxes.map((b) => ({ ...b })));
        this.selectedBoxId.set(null);
        this.itemStatus.set(det.status);
        this.noObjectsConfirmed.set(det.noObjectsConfirmed);
        this.reviewNotes.set(det.reviewNotes ?? '');
        this.fpRemoved.set(det.falsePositivesRemoved ?? 0);
        this.fnAdded.set(det.falseNegativesAdded ?? 0);
        this.undoStack.set([]);
        this.redoStack.set([]);
        this.dirty.set(false);
        if (this.classes().length > 0) {
          this.activeClassName.set(this.classes()[0]);
        }
      },
      error: (err) => {
        this.notifications.error(extractApiErrorMessage(err, 'Falha ao carregar item no editor.'));
      },
    });
  }

  protected onBoxCreated(partial: Omit<AnnotationBox, 'id'>): void {
    this.pushUndo();
    const id = `man_${Math.random().toString(36).slice(2, 10)}`;
    const nextBox: AnnotationBox = { ...partial, id };
    this.boxes.update((curr) => [...curr, nextBox]);
    this.selectedBoxId.set(id);
    this.noObjectsConfirmed.set(false);
    this.fnAdded.update((v) => v + 1);
    this.dirty.set(true);
    this.tool.set('select');
  }

  protected onBoxUpdated(updated: AnnotationBox): void {
    this.boxes.update((curr) => curr.map((b) => (b.id === updated.id ? updated : b)));
    this.dirty.set(true);
  }

  protected acceptBox(boxId: string): void {
    this.pushUndo();
    this.boxes.update((curr) =>
      curr.map((b) => (b.id === boxId ? { ...b, origin: 'ai_accepted' as const } : b))
    );
    this.dirty.set(true);
  }

  protected acceptAllAiSuggestions(): void {
    this.pushUndo();
    this.boxes.update((curr) =>
      curr.map((b) => (b.origin === 'ai_suggestion' ? { ...b, origin: 'ai_accepted' as const } : b))
    );
    this.dirty.set(true);
  }

  protected deleteSelectedBox(): void {
    const id = this.selectedBoxId();
    if (!id) return;
    this.removeBoxById(id);
  }

  protected removeBoxById(boxId: string): void {
    const target = this.boxes().find((b) => b.id === boxId);
    if (!target) return;
    this.pushUndo();
    if (target.origin === 'ai_suggestion') {
      this.fpRemoved.update((v) => v + 1);
    }
    this.boxes.update((curr) => curr.filter((b) => b.id !== boxId));
    if (this.selectedBoxId() === boxId) {
      this.selectedBoxId.set(null);
    }
    this.dirty.set(true);
  }

  protected toggleNoObjects(): void {
    const next = !this.noObjectsConfirmed();
    this.noObjectsConfirmed.set(next);
    if (next && this.boxes().length > 0) {
      this.pushUndo();
      const aiCount = this.boxes().filter((b) => b.origin === 'ai_suggestion').length;
      this.fpRemoved.update((v) => v + aiCount);
      this.boxes.set([]);
      this.selectedBoxId.set(null);
    }
    this.dirty.set(true);
  }

  protected onStatusChange(status: AnnotationItemStatus): void {
    this.itemStatus.set(status);
    this.dirty.set(true);
  }

  protected undo(): void {
    const stack = this.undoStack();
    if (stack.length === 0) return;
    const prev = stack[stack.length - 1];
    this.redoStack.update((r) => [...r, this.boxes().map((b) => ({ ...b }))]);
    this.undoStack.set(stack.slice(0, -1));
    this.boxes.set(prev);
    this.dirty.set(true);
  }

  protected redo(): void {
    const stack = this.redoStack();
    if (stack.length === 0) return;
    const next = stack[stack.length - 1];
    this.undoStack.update((u) => [...u, this.boxes().map((b) => ({ ...b }))]);
    this.redoStack.set(stack.slice(0, -1));
    this.boxes.set(next);
    this.dirty.set(true);
  }

  protected runSinglePreannotate(): void {
    const activeCp = this.checkpoints().find((c) => c.status === 'active') ?? this.checkpoints()[0];
    this.preannotating.set(true);
    this.api
      .preannotateItem(this.projectId(), this.itemId(), {
        modelId: activeCp?.id ?? 'exp04c_yolo26n_opentop_640',
        overwrite: true,
      })
      .subscribe({
        next: (det) => {
          this.preannotating.set(false);
          this.detail.set(det);
          this.boxes.set(det.boxes.map((b) => ({ ...b })));
          this.itemStatus.set(det.status);
          this.dirty.set(false);
          this.notifications.success(`${det.boxes.length} sugestões de IA geradas.`);
          this.itemSaved.emit();
        },
        error: (err) => {
          this.preannotating.set(false);
          this.notifications.error(extractApiErrorMessage(err, 'Falha na pré-anotação.'));
        },
      });
  }

  protected saveCurrent(explicitStatus?: AnnotationItemStatus): void {
    const resolvedStatus: AnnotationItemStatus =
      explicitStatus ??
      (this.itemStatus() === 'pending' || this.itemStatus() === 'pre_annotated'
        ? 'reviewed'
        : this.itemStatus());

    const normalizedBoxes: AnnotationBox[] = this.boxes().map((b) =>
      b.origin === 'ai_suggestion' && resolvedStatus === 'approved'
        ? { ...b, origin: 'ai_accepted' }
        : b
    );

    this.saving.set(true);
    this.api
      .updateItemAnnotations(this.projectId(), this.itemId(), {
        boxes: normalizedBoxes,
        noObjectsConfirmed: this.noObjectsConfirmed(),
        status: resolvedStatus,
        reviewNotes: this.reviewNotes() || null,
        falsePositivesRemoved: this.fpRemoved(),
        falseNegativesAdded: this.fnAdded(),
      })
      .subscribe({
        next: (saved) => {
          this.saving.set(false);
          this.detail.set(saved);
          this.boxes.set(saved.boxes);
          this.itemStatus.set(saved.status);
          this.dirty.set(false);
          this.notifications.success(
            resolvedStatus === 'approved' ? 'Revisão aprovada e salva.' : 'Anotações salvas.'
          );
          this.itemSaved.emit();
        },
        error: (err) => {
          this.saving.set(false);
          this.notifications.error(extractApiErrorMessage(err, 'Falha ao salvar anotações.'));
        },
      });
  }

  protected navigateBy(delta: number): void {
    const nextIdx = this.currentIndex() + delta;
    const list = this.items();
    if (nextIdx >= 0 && nextIdx < list.length) {
      this.itemSelected.emit(list[nextIdx].id);
    }
  }

  protected confirmExit(): void {
    if (this.dirty()) {
      this.saveCurrent();
    }
    this.exitEditor.emit();
  }

  private pushUndo(): void {
    const snapshot = this.boxes().map((b) => ({ ...b }));
    this.undoStack.update((s) => [...s.slice(-24), snapshot]);
    this.redoStack.set([]);
  }
}

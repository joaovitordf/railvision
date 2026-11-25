import { Component, DestroyRef, OnDestroy, OnInit, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { catchError, distinctUntilChanged, map, of } from 'rxjs';
import { ModelCheckpointSummary } from '../../../../core/models/runtime.models';
import { extractApiErrorMessage } from '../../../../core/services/api-error.util';
import { LayoutStateService } from '../../../../core/services/layout-state.service';
import { RuntimeStatusService } from '../../../../core/services/runtime-status.service';
import { AnnotationEditorComponent } from '../../components/annotation-editor.component';
import { AnnotationMediaPanelComponent } from '../../components/annotation-media-panel.component';
import { AnnotationProjectDetail } from '../../models/annotations.models';
import { AnnotationsApiService } from '../../services/annotations-api.service';

@Component({
  selector: 'rv-annotation-project-page',
  standalone: true,
  imports: [
    RouterLink,
    MatIconModule,
    MatProgressBarModule,
    AnnotationMediaPanelComponent,
    AnnotationEditorComponent,
  ],
  template: `
    @if (editorItemId(); as activeItemId) {
      @if (project(); as proj) {
        <rv-annotation-editor
          [projectId]="proj.id"
          [items]="proj.items"
          [itemId]="activeItemId"
          [classes]="proj.classes"
          [checkpoints]="checkpoints()"
          (itemSelected)="editorItemId.set($event)"
          (itemSaved)="refreshProject(proj.id)"
          (exitEditor)="closeEditor()"
        />
      }
    } @else {
      <div class="rv-page">
        <div class="rv-page-header">
          <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
            <a class="rv-btn rv-btn--secondary" routerLink="/annotations">
              <mat-icon>arrow_back</mat-icon>
              <span>Projetos</span>
            </a>
            @if (project(); as proj) {
              <div>
                <h2 class="rv-page-header__title" style="font-size: 1.3rem;">{{ proj.name }}</h2>
                <div class="rv-mono" style="font-size: 0.78rem; color: var(--rv-text-muted); margin-top: 2px;">
                  {{ proj.campaignId }} · {{ proj.sourceOrigin }} · {{ proj.reviewedCount }}/{{ proj.itemCount }} revisados
                </div>
              </div>
            }
          </div>

          @if (project(); as proj) {
            <button
              type="button"
              class="rv-btn rv-btn--primary"
              data-testid="open-studio-editor-btn"
              [disabled]="proj.items.length === 0"
              (click)="openEditor(proj.items[0].id)"
            >
              <mat-icon>crop_free</mat-icon>
              <span>Abrir editor ({{ proj.items.length }})</span>
            </button>
          }
        </div>

        @if (loading()) {
          <mat-progress-bar mode="indeterminate" />
        }

        @if (errorMessage()) {
          <div class="rv-banner rv-banner--error" role="alert">
            <mat-icon aria-hidden="true">error_outline</mat-icon>
            <div>{{ errorMessage() }}</div>
          </div>
        } @else if (project(); as proj) {
          <rv-annotation-media-panel
            [project]="proj"
            [checkpoints]="checkpoints()"
            (projectUpdated)="refreshProject(proj.id)"
            (openInEditor)="openEditor($event)"
          />
        }
      </div>
    }
  `,
})
export class AnnotationProjectPageComponent implements OnInit, OnDestroy {
  private readonly api = inject(AnnotationsApiService);
  private readonly route = inject(ActivatedRoute);
  private readonly layout = inject(LayoutStateService);
  private readonly runtimeStatus = inject(RuntimeStatusService, { optional: true });
  private readonly destroyRef = inject(DestroyRef);

  protected readonly project = signal<AnnotationProjectDetail | null>(null);
  protected readonly checkpoints = signal<ModelCheckpointSummary[]>([]);
  protected readonly loading = signal<boolean>(false);
  protected readonly errorMessage = signal<string | null>(null);
  protected readonly editorItemId = signal<string | null>(null);

  ngOnInit(): void {
    if (this.runtimeStatus) {
      this.runtimeStatus
        .listCheckpoints()
        .pipe(catchError(() => of([] as ModelCheckpointSummary[])), takeUntilDestroyed(this.destroyRef))
        .subscribe((cps) => this.checkpoints.set(cps));
    }

    this.route.paramMap
      .pipe(
        map((p) => p.get('projectId') || ''),
        distinctUntilChanged(),
        takeUntilDestroyed(this.destroyRef)
      )
      .subscribe((pid) => {
        if (pid) {
          this.closeEditor();
          this.loadProject(pid);
        }
      });
  }

  ngOnDestroy(): void {
    this.layout.setFullBleed(false);
  }

  protected openEditor(itemId: string): void {
    this.editorItemId.set(itemId);
    this.layout.setFullBleed(true);
  }

  protected closeEditor(): void {
    this.editorItemId.set(null);
    this.layout.setFullBleed(false);
  }

  protected loadProject(projectId: string): void {
    this.loading.set(true);
    this.errorMessage.set(null);
    this.api
      .getProject(projectId)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (proj) => {
          this.project.set(proj);
          this.loading.set(false);
        },
        error: (err) => {
          this.loading.set(false);
          this.errorMessage.set(extractApiErrorMessage(err, 'Projeto de anotação não encontrado.'));
        },
      });
  }

  protected refreshProject(projectId: string): void {
    this.api
      .getProject(projectId)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (proj) => this.project.set(proj),
        error: (err) => {
          this.errorMessage.set(extractApiErrorMessage(err, 'Falha ao atualizar projeto de anotação.'));
        },
      });
  }
}

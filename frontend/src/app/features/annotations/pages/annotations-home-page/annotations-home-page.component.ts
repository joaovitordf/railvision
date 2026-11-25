import { DatePipe } from '@angular/common';
import { Component, OnInit, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { Router, RouterLink } from '@angular/router';
import { extractApiErrorMessage } from '../../../../core/services/api-error.util';
import { NotificationService } from '../../../../core/services/notification.service';
import { EmptyStateComponent } from '../../../../shared/components/empty-state/empty-state.component';
import { AnnotationProjectSummary } from '../../models/annotations.models';
import { AnnotationsApiService } from '../../services/annotations-api.service';

@Component({
  selector: 'rv-annotations-home-page',
  standalone: true,
  imports: [
    DatePipe,
    FormsModule,
    RouterLink,
    MatIconModule,
    MatProgressBarModule,
    EmptyStateComponent,
  ],
  template: `
    <div class="rv-page">
      <div class="rv-page-header">
        <div>
          <h2 class="rv-page-header__title">Anotações & Curadoria (Studio MVP)</h2>
          <p class="rv-page-header__subtitle">
            Projetos por campanha, extração deduplicada de frames de vídeo e revisão humana com pré-anotação IA.
          </p>
        </div>

        <div class="rv-header-actions">
          <button type="button" class="rv-btn rv-btn--secondary" (click)="loadProjects()" [disabled]="loading()">
            <mat-icon>refresh</mat-icon>
            <span>Atualizar</span>
          </button>
          <button
            type="button"
            class="rv-btn rv-btn--primary"
            data-testid="new-annotation-project-btn"
            (click)="showCreate.set(!showCreate())"
          >
            <mat-icon>{{ showCreate() ? 'close' : 'add' }}</mat-icon>
            <span>{{ showCreate() ? 'Fechar formulário' : 'Novo projeto' }}</span>
          </button>
        </div>
      </div>

      <div class="rv-banner rv-banner--info" style="align-items: center;">
        <mat-icon aria-hidden="true">info</mat-icon>
        <div style="font-size: 0.85rem;">
          <strong>Fluxo MVP de Imagens e Extração de Frames:</strong> classe ativa
          <span class="rv-mono">open_top_freight_wagon</span>. Aprovação humana nunca altera splits oficiais automaticamente
          e exporta pacotes compatíveis com <strong>CVAT YOLO 1.1</strong>.
        </div>
      </div>

      @if (showCreate()) {
        <section class="rv-card" data-testid="create-project-card">
          <div class="rv-card__header">
            <div>
              <h3 class="rv-card__title">Criar projeto de anotação</h3>
              <p class="rv-card__subtitle">Isole imagens e vídeos por campanha de origem antes de validar para dataset.</p>
            </div>
          </div>

          <form (ngSubmit)="createProject()" style="display: flex; flex-direction: column; gap: 14px;">
            <div class="rv-form-grid">
              <div class="rv-field">
                <label class="rv-label" for="ann-proj-name">Nome do projeto</label>
                <input
                  id="ann-proj-name"
                  class="rv-input"
                  name="name"
                  [ngModel]="name()"
                  (ngModelChange)="name.set($event)"
                  required
                  placeholder="Ex.: Pátio Tubarão - Manhã"
                />
              </div>

              <div class="rv-field">
                <label class="rv-label" for="ann-proj-camp">ID da campanha</label>
                <input
                  id="ann-proj-camp"
                  class="rv-input"
                  name="campaignId"
                  [ngModel]="campaignId()"
                  (ngModelChange)="campaignId.set($event)"
                  required
                  placeholder="Ex.: camp_patio_tubarao"
                />
              </div>

              <div class="rv-field">
                <label class="rv-label" for="ann-proj-origin">Origem</label>
                <input
                  id="ann-proj-origin"
                  class="rv-input"
                  name="sourceOrigin"
                  [ngModel]="sourceOrigin()"
                  (ngModelChange)="sourceOrigin.set($event)"
                  placeholder="proprio / publico / cvat"
                />
              </div>
            </div>

            <div class="rv-field">
              <label class="rv-label" for="ann-proj-desc">Descrição (opcional)</label>
              <input
                id="ann-proj-desc"
                class="rv-input"
                name="description"
                [ngModel]="description()"
                (ngModelChange)="description.set($event)"
                placeholder="Contexto de câmera, iluminação ou observações da campanha"
              />
            </div>

            <div>
              <button
                type="submit"
                class="rv-btn rv-btn--primary"
                [disabled]="creating() || name().trim().length < 2 || campaignId().trim().length < 2"
              >
                <mat-icon>create_new_folder</mat-icon>
                <span>{{ creating() ? 'Criando...' : 'Criar projeto' }}</span>
              </button>
            </div>
          </form>
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
      } @else if (projects().length === 0 && !loading()) {
        <section class="rv-card">
          <rv-empty-state
            icon="crop_free"
            title="Nenhum projeto de anotação criado"
            description="Crie um projeto acima para enviar imagens, extrair frames de vídeos e revisar bounding boxes."
          />
        </section>
      } @else if (projects().length > 0) {
        <section class="rv-card rv-card--table">
          <div class="rv-card__header">
            <div>
              <h3 class="rv-card__title">Projetos de anotação ({{ projects().length }})</h3>
              <p class="rv-card__subtitle">Campanhas locais isoladas com rastreabilidade de origem e revisão humana.</p>
            </div>
          </div>

          <div class="rv-table-wrap">
            <table class="rv-table">
              <thead>
                <tr>
                  <th>Projeto</th>
                  <th>Campanha / Origem</th>
                  <th>Itens</th>
                  <th>Revisados</th>
                  <th>Vídeos</th>
                  <th>Atualizado</th>
                  <th>Ações</th>
                </tr>
              </thead>
              <tbody>
                @for (proj of projects(); track proj.id) {
                  <tr>
                    <td>
                      <a
                        [routerLink]="['/annotations', proj.id]"
                        style="color: var(--rv-accent); font-weight: 600; text-decoration: none;"
                      >
                        {{ proj.name }}
                      </a>
                      @if (proj.description) {
                        <div style="font-size: 0.78rem; color: var(--rv-text-muted);">{{ proj.description }}</div>
                      }
                    </td>
                    <td class="rv-mono">{{ proj.campaignId }} · {{ proj.sourceOrigin }}</td>
                    <td>{{ proj.itemCount }}</td>
                    <td>
                      <span
                        class="rv-badge"
                        [class.rv-badge--success]="proj.reviewedCount > 0"
                        [class.rv-badge--neutral]="proj.reviewedCount === 0"
                      >
                        <span class="rv-badge__dot" aria-hidden="true"></span>
                        {{ proj.reviewedCount }} / {{ proj.itemCount }}
                      </span>
                    </td>
                    <td>{{ proj.videoCount }}</td>
                    <td>{{ proj.updatedAt | date: 'dd/MM/yyyy HH:mm' }}</td>
                    <td>
                      <div style="display: flex; align-items: center; gap: 6px;">
                        <a class="rv-btn rv-btn--secondary rv-btn--sm" [routerLink]="['/annotations', proj.id]">
                          Abrir
                        </a>
                        <button
                          type="button"
                          class="rv-icon-btn"
                          (click)="deleteProject(proj)"
                          aria-label="Excluir projeto"
                          title="Excluir projeto"
                        >
                          <mat-icon>delete_outline</mat-icon>
                        </button>
                      </div>
                    </td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
        </section>
      }
    </div>
  `,
})
export class AnnotationsHomePageComponent implements OnInit {
  private readonly api = inject(AnnotationsApiService);
  private readonly notifications = inject(NotificationService);
  private readonly router = inject(Router);

  protected readonly projects = signal<AnnotationProjectSummary[]>([]);
  protected readonly loading = signal<boolean>(false);
  protected readonly creating = signal<boolean>(false);
  protected readonly showCreate = signal<boolean>(false);
  protected readonly errorMessage = signal<string | null>(null);

  protected readonly name = signal<string>('');
  protected readonly campaignId = signal<string>('');
  protected readonly sourceOrigin = signal<string>('proprio');
  protected readonly description = signal<string>('');

  ngOnInit(): void {
    this.loadProjects();
  }

  protected loadProjects(): void {
    this.loading.set(true);
    this.errorMessage.set(null);
    this.api.listProjects().subscribe({
      next: (list) => {
        this.projects.set(list);
        this.loading.set(false);
      },
      error: (err) => {
        this.loading.set(false);
        this.errorMessage.set(extractApiErrorMessage(err, 'Falha ao listar projetos de anotação.'));
      },
    });
  }

  protected createProject(): void {
    this.creating.set(true);
    this.api
      .createProject({
        name: this.name().trim(),
        campaignId: this.campaignId().trim(),
        sourceOrigin: this.sourceOrigin().trim() || 'proprio',
        description: this.description().trim() || null,
      })
      .subscribe({
        next: (created) => {
          this.creating.set(false);
          this.notifications.success(`Projeto '${created.name}' criado.`);
          this.router.navigate(['/annotations', created.id]);
        },
        error: (err) => {
          this.creating.set(false);
          this.notifications.error(extractApiErrorMessage(err, 'Falha ao criar projeto.'));
        },
      });
  }

  protected deleteProject(proj: AnnotationProjectSummary): void {
    this.api.deleteProject(proj.id).subscribe({
      next: () => {
        this.notifications.info(`Projeto '${proj.name}' removido.`);
        this.loadProjects();
      },
      error: (err) => {
        this.notifications.error(extractApiErrorMessage(err, 'Falha ao excluir projeto.'));
      },
    });
  }
}

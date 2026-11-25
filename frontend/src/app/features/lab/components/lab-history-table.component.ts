import { DatePipe, DecimalPipe } from '@angular/common';
import { Component, input, output } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { EmptyStateComponent } from '../../../shared/components/empty-state/empty-state.component';
import { StatusBadgeComponent } from '../../../shared/components/status-badge/status-badge.component';
import { RunListItem } from '../models/lab.models';

@Component({
  selector: 'rv-lab-history-table',
  standalone: true,
  imports: [DatePipe, DecimalPipe, MatIconModule, EmptyStateComponent, StatusBadgeComponent],
  template: `
    <section class="rv-card rv-card--table">
      <div class="rv-card__header">
        <div>
          <h3 class="rv-card__title">Histórico de execuções no Laboratório</h3>
          <p class="rv-card__subtitle">
            Clique em qualquer execução para abrir resultados detalhados e frames na aba Inferência.
          </p>
        </div>
        <button type="button" class="rv-btn rv-btn--secondary rv-btn--sm" (click)="refreshRequested.emit()">
          <mat-icon>refresh</mat-icon>
          <span>Atualizar</span>
        </button>
      </div>

      @if (runsError()) {
        <div style="padding: 16px 22px 0;">
          <div class="rv-banner rv-banner--error" role="alert">
            <mat-icon aria-hidden="true">error_outline</mat-icon>
            <div>{{ runsError() }}</div>
          </div>
        </div>
      }

      @if (runs().length === 0) {
        <rv-empty-state
          icon="science"
          title="Nenhum teste executado ainda"
          description="Envie uma imagem ou vídeo na aba Inferência para executar e registrar testes aqui."
        />
      } @else {
        <div class="rv-table-wrap">
          <table class="rv-table">
            <thead>
              <tr>
                <th>Arquivo</th>
                <th>Modelo</th>
                <th>Tipo</th>
                <th>Status</th>
                <th>Detecções</th>
                <th>FPS / Latência</th>
                <th>Dispositivo</th>
                <th>Data</th>
              </tr>
            </thead>
            <tbody>
              @for (item of runs(); track item.runId) {
                <tr
                  class="rv-row--selectable"
                  [class.rv-row--selected]="activeRunId() === item.runId"
                  (click)="runSelected.emit(item.runId)"
                >
                  <td><strong>{{ item.originalFilename }}</strong></td>
                  <td class="rv-mono">{{ item.modelName }}</td>
                  <td>{{ item.mediaType === 'IMAGE' ? 'Imagem' : 'Vídeo' }}</td>
                  <td><rv-status-badge [status]="item.status" /></td>
                  <td>{{ item.noDetections ? '0 (nenhum)' : item.totalDetections }}</td>
                  <td>
                    @if (item.processingFps != null) {
                      {{ item.processingFps | number: '1.1-1' }} FPS ({{ item.avgFrameMs | number: '1.1-1' }} ms)
                    } @else {
                      —
                    }
                  </td>
                  <td class="rv-mono">{{ item.deviceType }} ({{ item.activeExecutionProvider }})</td>
                  <td>{{ item.createdAt | date: 'dd/MM HH:mm:ss' }}</td>
                </tr>
              }
            </tbody>
          </table>
        </div>
      }
    </section>
  `,
})
export class LabHistoryTableComponent {
  readonly runs = input<RunListItem[]>([]);
  readonly activeRunId = input<string | null>(null);
  readonly runsError = input<string | null>(null);

  readonly refreshRequested = output<void>();
  readonly runSelected = output<string>();
}

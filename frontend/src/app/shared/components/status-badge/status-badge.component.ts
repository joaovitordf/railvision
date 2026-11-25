import { Component, computed, input } from '@angular/core';
import { InspectionStatus, JobStatus } from '../../../core/models/inspection.models';

type AnyStatus = InspectionStatus | JobStatus | string;

@Component({
  selector: 'rv-status-badge',
  standalone: true,
  template: `
    <span [class]="'rv-badge ' + badgeClass()">
      <span class="rv-badge__dot" aria-hidden="true"></span>
      <span>{{ label() }}</span>
    </span>
  `,
})
export class StatusBadgeComponent {
  readonly status = input.required<AnyStatus>();

  protected readonly label = computed(() => {
    switch (this.status()) {
      case 'UPLOADED':
        return 'Enviado';
      case 'QUEUED':
        return 'Na fila';
      case 'PROCESSING':
      case 'RUNNING':
        return 'Processando';
      case 'COMPLETED':
        return 'Concluído';
      case 'FAILED':
        return 'Falhou';
      default:
        return this.status();
    }
  });

  protected readonly badgeClass = computed(() => {
    switch (this.status()) {
      case 'COMPLETED':
        return 'rv-badge--success';
      case 'QUEUED':
      case 'PROCESSING':
      case 'RUNNING':
        return 'rv-badge--info';
      case 'FAILED':
        return 'rv-badge--error';
      default:
        return 'rv-badge--neutral';
    }
  });
}

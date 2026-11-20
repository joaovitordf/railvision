import { Component, input } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'rv-empty-state',
  standalone: true,
  imports: [MatIconModule],
  template: `
    <div
      style="text-align: center; padding: 36px 16px; color: var(--rv-text-secondary); display: flex; flex-direction: column; align-items: center; gap: 8px;"
    >
      <mat-icon aria-hidden="true" style="font-size: 40px; width: 40px; height: 40px; color: var(--rv-text-muted);">
        {{ icon() }}
      </mat-icon>
      <strong style="color: var(--rv-text-primary); font-size: 1rem;">{{ title() }}</strong>
      @if (description()) {
        <span style="max-width: 480px; font-size: 0.9rem;">{{ description() }}</span>
      }
    </div>
  `,
})
export class EmptyStateComponent {
  readonly icon = input<string>('inbox');
  readonly title = input.required<string>();
  readonly description = input<string>('');
}

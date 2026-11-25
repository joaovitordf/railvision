import { Component, input } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'rv-empty-state',
  standalone: true,
  imports: [MatIconModule],
  template: `
    <div class="rv-empty-state">
      <div class="rv-empty-state__icon">
        <mat-icon aria-hidden="true">{{ icon() }}</mat-icon>
      </div>
      <strong class="rv-empty-state__title">{{ title() }}</strong>
      @if (description()) {
        <span class="rv-empty-state__desc">{{ description() }}</span>
      }
    </div>
  `,
})
export class EmptyStateComponent {
  readonly icon = input<string>('inbox');
  readonly title = input.required<string>();
  readonly description = input<string>('');
}

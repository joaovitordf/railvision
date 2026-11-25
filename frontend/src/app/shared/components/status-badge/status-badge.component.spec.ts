import { TestBed } from '@angular/core/testing';
import { StatusBadgeComponent } from './status-badge.component';

describe('StatusBadgeComponent', () => {
  it('retains base rv-badge class alongside status modifier class and status dot', () => {
    const fixture = TestBed.createComponent(StatusBadgeComponent);
    fixture.componentRef.setInput('status', 'COMPLETED');
    fixture.detectChanges();

    const pill = fixture.nativeElement.querySelector('span.rv-badge') as HTMLElement | null;
    expect(pill).toBeTruthy();
    expect(pill?.classList.contains('rv-badge')).toBe(true);
    expect(pill?.classList.contains('rv-badge--success')).toBe(true);
    expect(pill?.querySelector('.rv-badge__dot')).toBeTruthy();
    expect(pill?.textContent?.trim()).toBe('Concluído');
  });
});

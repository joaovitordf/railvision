import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';
import { vi } from 'vitest';
import { NotificationService } from '../../../../core/services/notification.service';
import { AnnotationsApiService } from '../../services/annotations-api.service';
import { AnnotationsHomePageComponent } from './annotations-home-page.component';

describe('AnnotationsHomePageComponent', () => {
  it('lists annotation projects with reviewed counts and scope notice', async () => {
    const apiMock = {
      listProjects: vi.fn().mockReturnValue(
        of([
          {
            id: 'proj-1',
            name: 'Campanha Pátio Sul',
            campaignId: 'camp_sul_01',
            sourceOrigin: 'proprio',
            description: 'Revisão inicial',
            classes: ['open_top_freight_wagon'],
            itemCount: 10,
            videoCount: 1,
            statusCounts: { approved: 4, pre_annotated: 6 },
            reviewedCount: 4,
            approvedCount: 4,
            createdAt: '2026-11-24T00:00:00Z',
            updatedAt: '2026-11-24T01:00:00Z',
          },
        ])
      ),
      createProject: vi.fn(),
      deleteProject: vi.fn(),
    };

    await TestBed.configureTestingModule({
      imports: [AnnotationsHomePageComponent],
      providers: [
        provideRouter([]),
        { provide: AnnotationsApiService, useValue: apiMock },
        { provide: NotificationService, useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn() } },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(AnnotationsHomePageComponent);
    await fixture.whenStable();
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('Campanha Pátio Sul');
    expect(el.textContent).toContain('4 / 10');
    expect(el.textContent).toContain('open_top_freight_wagon');
  });
});

import { Component, OnInit, inject } from '@angular/core';
import { Router, Routes } from '@angular/router';
import { ShellComponent } from './core/layout/shell/shell.component';

@Component({
  selector: 'rv-reports-redirect',
  standalone: true,
  template: '',
})
export class ReportsRedirectComponent implements OnInit {
  private readonly router = inject(Router);

  ngOnInit(): void {
    this.router.navigate(['/inspections'], {
      queryParams: { tab: 'reports' },
      replaceUrl: true,
    });
  }
}

export const routes: Routes = [
  {
    path: '',
    component: ShellComponent,
    children: [
      {
        path: 'overview',
        loadChildren: () =>
          import('./features/overview/overview.routes').then((m) => m.OVERVIEW_ROUTES),
      },
      {
        path: 'inspections/:id/report',
        loadChildren: () =>
          import('./features/inspection-details/inspection-details.routes').then(
            (m) => m.INSPECTION_DETAILS_ROUTES
          ),
        data: { initialTab: 'report' },
      },
      {
        path: 'inspections/:id',
        loadChildren: () =>
          import('./features/inspection-details/inspection-details.routes').then(
            (m) => m.INSPECTION_DETAILS_ROUTES
          ),
      },
      {
        path: 'inspections',
        loadChildren: () =>
          import('./features/inspections/inspections.routes').then((m) => m.INSPECTIONS_ROUTES),
      },
      {
        path: 'reports',
        component: ReportsRedirectComponent,
      },
      {
        path: 'annotations',
        loadChildren: () =>
          import('./features/annotations/annotations.routes').then((m) => m.ANNOTATIONS_ROUTES),
      },
      {
        path: 'lab',
        loadChildren: () =>
          import('./features/lab/lab.routes').then((m) => m.LAB_ROUTES),
      },
      {
        path: 'vision-lab',
        pathMatch: 'full',
        redirectTo: 'lab',
      },
      {
        path: 'onnx-lab',
        pathMatch: 'full',
        redirectTo: 'lab',
      },
      {
        path: '',
        pathMatch: 'full',
        redirectTo: 'overview',
      },
    ],
  },
  {
    path: '**',
    redirectTo: 'overview',
  },
];

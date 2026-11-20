import { Routes } from '@angular/router';
import { ShellComponent } from './core/layout/shell/shell.component';

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
        loadChildren: () =>
          import('./features/reports/reports.routes').then((m) => m.REPORTS_ROUTES),
      },
      {
        path: 'onnx-lab',
        loadChildren: () =>
          import('./features/onnx-lab/onnx-lab.routes').then((m) => m.ONNX_LAB_ROUTES),
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

import { Routes } from '@angular/router';
import { OverviewPageComponent } from './pages/overview-page/overview-page.component';

export const OVERVIEW_ROUTES: Routes = [
  {
    path: '',
    component: OverviewPageComponent,
    data: { title: 'Visão geral', icon: 'dashboard' },
  },
];

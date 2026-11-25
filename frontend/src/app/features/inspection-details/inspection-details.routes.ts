import { Routes } from '@angular/router';
import { InspectionDetailPageComponent } from './pages/inspection-detail-page/inspection-detail-page.component';

export const INSPECTION_DETAILS_ROUTES: Routes = [
  {
    path: '',
    component: InspectionDetailPageComponent,
    data: { title: 'Detalhe da inspeção', icon: 'train' },
  },
];

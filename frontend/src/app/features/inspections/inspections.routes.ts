import { Routes } from '@angular/router';
import { InspectionsPageComponent } from './pages/inspections-page/inspections-page.component';

export const INSPECTIONS_ROUTES: Routes = [
  {
    path: '',
    component: InspectionsPageComponent,
    data: { title: 'Inspeções', icon: 'train' },
  },
];

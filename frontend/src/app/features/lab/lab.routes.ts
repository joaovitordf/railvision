import { Routes } from '@angular/router';
import { LabPageComponent } from './pages/lab-page/lab-page.component';

export const LAB_ROUTES: Routes = [
  {
    path: 'training',
    component: LabPageComponent,
    data: { title: 'Laboratório · Treinamento', icon: 'trending_up', initialTab: 'training' },
  },
  {
    path: 'models',
    component: LabPageComponent,
    data: { title: 'Laboratório · Modelos', icon: 'memory', initialTab: 'models' },
  },
  {
    path: 'runs/:runId',
    component: LabPageComponent,
    data: { title: 'Laboratório', icon: 'science', initialTab: 'inference' },
  },
  {
    path: '',
    component: LabPageComponent,
    data: { title: 'Laboratório', icon: 'science', initialTab: 'inference' },
  },
];

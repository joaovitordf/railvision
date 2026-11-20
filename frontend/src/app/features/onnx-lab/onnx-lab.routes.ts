import { Routes } from '@angular/router';
import { OnnxLabPageComponent } from './pages/onnx-lab-page/onnx-lab-page.component';

export const ONNX_LAB_ROUTES: Routes = [
  {
    path: '',
    component: OnnxLabPageComponent,
    data: { title: 'Laboratório de Visão', icon: 'biotech' },
  },
];

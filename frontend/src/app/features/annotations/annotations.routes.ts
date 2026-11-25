import { Routes } from '@angular/router';
import { AnnotationProjectPageComponent } from './pages/annotation-project-page/annotation-project-page.component';
import { AnnotationsHomePageComponent } from './pages/annotations-home-page/annotations-home-page.component';

export const ANNOTATIONS_ROUTES: Routes = [
  {
    path: ':projectId',
    component: AnnotationProjectPageComponent,
    data: { title: 'Anotações · Studio', icon: 'crop_free' },
  },
  {
    path: '',
    component: AnnotationsHomePageComponent,
    data: { title: 'Anotações', icon: 'crop_free' },
  },
];

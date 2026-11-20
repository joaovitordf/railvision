export interface NavigationItem {
  label: string;
  route: string;
  icon: string;
}

export const NAVIGATION_ITEMS: NavigationItem[] = [
  {
    label: 'Visão geral',
    route: '/overview',
    icon: 'dashboard',
  },
  {
    label: 'Inspeções',
    route: '/inspections',
    icon: 'train',
  },
  {
    label: 'Relatórios',
    route: '/reports',
    icon: 'summarize',
  },
  {
    label: 'Laboratório de Visão',
    route: '/onnx-lab',
    icon: 'biotech',
  },
];

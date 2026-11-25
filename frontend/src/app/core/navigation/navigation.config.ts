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
    label: 'Anotações',
    route: '/annotations',
    icon: 'crop_free',
  },
  {
    label: 'Laboratório',
    route: '/lab',
    icon: 'science',
  },
];

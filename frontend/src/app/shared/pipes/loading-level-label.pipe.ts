import { Pipe, PipeTransform } from '@angular/core';

const LABELS: Record<string, string> = {
  empty: 'Vazio',
  low: 'Baixo',
  medium: 'Médio',
  high: 'Alto',
  very_high: 'Muito alto',
  unknown: 'Desconhecido',
  not_evaluated: 'Não avaliado (modelo pendente)',
};

@Pipe({
  name: 'loadingLevelLabel',
  standalone: true,
})
export class LoadingLevelLabelPipe implements PipeTransform {
  transform(level: string | null | undefined): string {
    if (!level) {
      return 'Não informado';
    }
    return LABELS[level] ?? level;
  }
}

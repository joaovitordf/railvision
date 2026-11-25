import { Pipe, PipeTransform } from '@angular/core';

const LABELS: Record<string, string> = {
  iron_ore: 'Minério de ferro',
  coal: 'Carvão',
  limestone: 'Calcário',
  sand: 'Areia',
  other_material: 'Outro material',
  unknown: 'Desconhecido',
  not_evaluated: 'Não avaliado (modelo pendente)',
};

@Pipe({
  name: 'materialLabel',
  standalone: true,
})
export class MaterialLabelPipe implements PipeTransform {
  transform(materialClass: string | null | undefined, loadingLevel?: string | null): string {
    if (materialClass == null || materialClass === '') {
      if (loadingLevel && loadingLevel !== 'empty') {
        return 'Não avaliado';
      }
      return 'N/A (vagão vazio)';
    }
    return LABELS[materialClass] ?? materialClass;
  }
}

import { LoadingLevelLabelPipe } from './loading-level-label.pipe';
import { MaterialLabelPipe } from './material-label.pipe';

describe('Shared Domain Label Pipes', () => {
  it('translates material classes and maps null to N/A (vagão vazio)', () => {
    const pipe = new MaterialLabelPipe();
    expect(pipe.transform(null)).toBe('N/A (vagão vazio)');
    expect(pipe.transform(undefined)).toBe('N/A (vagão vazio)');
    expect(pipe.transform('iron_ore')).toBe('Minério de ferro');
    expect(pipe.transform('coal')).toBe('Carvão');
    expect(pipe.transform('unknown')).toBe('Desconhecido');
    expect(pipe.transform('other_material')).toBe('Outro material');
    expect(pipe.transform('not_evaluated')).toBe('Não avaliado (modelo pendente)');
  });

  it('translates ordinal loading levels into pt-BR', () => {
    const pipe = new LoadingLevelLabelPipe();
    expect(pipe.transform('empty')).toBe('Vazio');
    expect(pipe.transform('low')).toBe('Baixo');
    expect(pipe.transform('medium')).toBe('Médio');
    expect(pipe.transform('high')).toBe('Alto');
    expect(pipe.transform('very_high')).toBe('Muito alto');
    expect(pipe.transform('not_evaluated')).toBe('Não avaliado (modelo pendente)');
  });
});

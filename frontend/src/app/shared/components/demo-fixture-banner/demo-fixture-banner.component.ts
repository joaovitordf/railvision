import { Component, input } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'rv-demo-fixture-banner',
  standalone: true,
  imports: [MatIconModule],
  template: `
    <div class="rv-banner rv-banner--warning" role="status" data-testid="demo-fixture-banner">
      <mat-icon aria-hidden="true">science</mat-icon>
      <div>
        <strong>Registro histórico de demonstração (dados fictícios)</strong>
        <div>
          Este registro foi gerado anteriormente pelo motor de exemplo
          <span class="rv-mono">{{ engineType() }}</span> (<span class="rv-mono">isExample=true</span>).
          Reprocesse a inspeção para executar a inferência real em PyTorch CUDA + ByteTrack.
        </div>
      </div>
    </div>
  `,
})
export class DemoFixtureBannerComponent {
  readonly engineType = input<string>('DEMO_FIXTURE_ENGINE');
}

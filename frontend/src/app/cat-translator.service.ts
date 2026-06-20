import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../environments/environment';

export interface PredictionScore {
  label: string;
  confidence: number;
}

export interface CatTranslationResponse {
  intent: string;
  top_guess: string;
  confidence: number;
  all_predictions: PredictionScore[];
  message: string;
  disclaimer?: string;
}

@Injectable({
  providedIn: 'root',
})
export class CatTranslatorService {
  private readonly apiUrl = `${environment.apiBaseUrl}/translate`;

  constructor(private readonly http: HttpClient) {}

  translateAudio(audioBlob: Blob): Observable<CatTranslationResponse> {
    const formData = new FormData();
    formData.append('file', audioBlob, 'cat-recording.webm');

    return this.http.post<CatTranslationResponse>(this.apiUrl, formData);
  }
}

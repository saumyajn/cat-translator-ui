import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

export const CAT_TRANSLATOR_API_URL = 'http://localhost:8000/translate';

export interface PredictionScore {
  label: string;
  confidence: number;
}

export interface CatTranslationResponse {
  intent: string;
  confidence: number;
  all_predictions: PredictionScore[];
  message: string;
  disclaimer?: string;
}

@Injectable({
  providedIn: 'root',
})
export class CatTranslatorService {
  private readonly apiUrl = CAT_TRANSLATOR_API_URL;

  constructor(private readonly http: HttpClient) {}

  translateAudio(audioBlob: Blob): Observable<CatTranslationResponse> {
    const formData = new FormData();
    formData.append('file', audioBlob, 'cat-recording.webm');

    return this.http.post<CatTranslationResponse>(this.apiUrl, formData);
  }
}

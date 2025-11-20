import { Status } from "../../../@types/log";
import { LogDetails } from "../../../client/api/types";

export interface SampleRow {
  logFile: string;
  task: string;
  model: string;
  status?: Status;
  sampleId: string | number;
  epoch: number;
  input: string;
  target: string;
  error?: string;
  limit?: string;
  retries?: number;
  completed?: boolean;
  evalSetId?: string;
  [key: string]: any;
}

export interface SamplesDataProvider {
  getSamples: () => SampleRow[];
  getLogDetails: () => Record<string, LogDetails>;
  isLoading: () => boolean;
  getTotalCount?: () => number;
}

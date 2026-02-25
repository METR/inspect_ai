import Dexie from "dexie";
import { EvalSample } from "../../@types/log";
import { LogDetails, LogPreview } from "../api/types";

// Logs Table - Basic file listing
export interface LogHandleRecord {
  // Auto-incrementing primary key for insertion order
  id?: number;
  file_path: string;
  file_name: string;
  task?: string;
  task_id?: string;
  mtime?: number;
  cached_at: string;
}

// Log Previews Table - Stores results from get_log_summaries()
export interface LogPreviewRecord {
  // Primary key
  file_path: string;

  // The complete log summary object
  preview: LogPreview;

  cached_at: string;
}

// Log Details Table - Stores complete results from get_log_info()
// This includes the full header and sample summaries
export interface LogDetailsRecord {
  // Primary key
  file_path: string;

  // The complete log info object (includes sample summaries)
  details: LogDetails;

  cached_at: string;
}

// Log Samples Table - Caches parsed samples to avoid re-downloading from S3
export interface LogSampleRecord {
  file_path: string;
  sample_id: string;
  epoch: number;
  file_mtime: number;
  cached_at: string;
  sample: EvalSample;
}

// Current database schema version
export const DB_VERSION = 10;

// Resolves a log dir into a database name
function resolveDBName(databaseHandle: string): string {
  const sanitizedDir = databaseHandle.replace(/[^a-zA-Z0-9_-]/g, "_");
  const dbName = `InspectAI_${sanitizedDir}`;
  return dbName;
}

export class AppDatabase extends Dexie {
  logs!: Dexie.Table<LogHandleRecord, number>;
  log_previews!: Dexie.Table<LogPreviewRecord, string>;
  log_details!: Dexie.Table<LogDetailsRecord, string>;
  log_samples!: Dexie.Table<LogSampleRecord, [string, string, number]>;

  /**
   * Check if an existing database needs to be recreated due to version mismatch.
   * Returns true if the database should be deleted and recreated.
   */
  static async checkVersionMismatch(databaseHandle: string): Promise<boolean> {
    const dbName = resolveDBName(databaseHandle);

    try {
      // Use indexedDB.databases() to check version without opening a connection.
      // Opening a temp Dexie connection and closing it leaves the underlying IDB
      // connection open briefly, which blocks Dexie.delete() from succeeding.
      const databases = await indexedDB.databases();
      const existing = databases.find((db) => db.name === dbName);
      if (!existing) {
        return false;
      }

      // Dexie stores IndexedDB version as dexieVersion * 10
      const dexieVersion = (existing.version || 0) / 10;
      if (dexieVersion !== DB_VERSION) {
        console.log(
          `Database version mismatch (found v${dexieVersion}, expected v${DB_VERSION})`,
        );
        return true;
      }
      return false;
    } catch (error) {
      // Database doesn't exist or has issues - let normal flow handle it
      return false;
    }
  }

  constructor(databaseHandle: string) {
    super(resolveDBName(databaseHandle));

    this.version(DB_VERSION).stores({
      // Basic file listing - indexes for querying and sorting
      logs: "++id, &file_path, mtime, task, task_id, cached_at",

      // Log summaries from get_log_summaries() - indexes for common queries
      log_previews:
        "file_path, preview.status, preview.task_id, preview.model, cached_at",

      // Complete log info from get_log_details() - includes samples
      log_details: "file_path, details.status, cached_at",

      // Cached parsed samples - avoids re-downloading from S3
      log_samples: "[file_path+sample_id+epoch], file_path, cached_at",
    });
  }
}

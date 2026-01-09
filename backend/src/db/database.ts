import Database from "better-sqlite3";
import path from "path";
import fs from "fs";

const resolveDbPath = (): string => {
  if (process.env.DATABASE_PATH) {
    return path.resolve(process.env.DATABASE_PATH);
  }
  return path.resolve(__dirname, "../../data/app.db");
};

const databasePath = resolveDbPath();
fs.mkdirSync(path.dirname(databasePath), { recursive: true });

export const db = new Database(databasePath);
db.pragma("foreign_keys = ON");

export const runInTransaction = <T>(callback: () => T): T => {
  const transaction = db.transaction(callback);
  return transaction();
};

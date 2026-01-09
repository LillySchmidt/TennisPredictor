import "dotenv/config";
import express from "express";
import cors from "cors";
import { initializeDatabase } from "./db/init";
import { tournamentsRouter } from "./routes/tournaments";

const app = express();
const port = process.env.PORT ?? 3000;

app.use(cors());
app.use(express.json());

initializeDatabase();

app.get("/api/health", (_req, res) => {
  res.json({ status: "ok" });
});

app.use("/api/tournaments", tournamentsRouter);

app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  console.error(err);
  res.status(500).json({ message: err.message ?? "Internal Server Error" });
});

app.listen(port, () => {
  console.log(`Backend listening on port ${port}`);
});

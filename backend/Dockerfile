FROM node:22-alpine AS base
WORKDIR /app
COPY backend/package*.json ./
RUN npm install
COPY backend/tsconfig.json ./
COPY backend/src ./src
RUN npm run build && npm prune --omit=dev
COPY backend/data ./data
ENV PORT=3000
ENV DATABASE_PATH=/app/data/app.db
EXPOSE 3000
CMD ["node", "dist/index.js"]

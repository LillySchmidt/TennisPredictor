FROM node:22-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
ARG VITE_API_URL=http://localhost:15002/api
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app
RUN npm install -g serve
COPY --from=builder /app/dist ./dist
EXPOSE 15001
CMD ["serve", "-s", "dist", "-l", "15001", "--single"]

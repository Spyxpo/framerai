# FramerAI website image: builds the React app and serves it with nginx,
# proxying API and WebSocket traffic to the backend service.
FROM node:20-slim AS build

WORKDIR /app
COPY website/package*.json ./
RUN npm ci
COPY website/ ./
RUN npm run build

FROM nginx:alpine
COPY website/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

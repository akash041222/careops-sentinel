import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { ErrorBoundary, ToastProvider } from "./components/ui";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <ErrorBoundary><ToastProvider><App /></ToastProvider></ErrorBoundary>
);

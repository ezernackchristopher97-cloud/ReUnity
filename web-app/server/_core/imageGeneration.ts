/**
 * Image generation helper supporting both OpenAI DALL-E 3 and Manus Forge
 *
 * Example usage:
 *   const { url: imageUrl } = await generateImage({
 *     prompt: "A serene landscape with mountains"
 *   });
 *
 * For editing (OpenAI does not support direct editing, will generate new image):
 *   const { url: imageUrl } = await generateImage({
 *     prompt: "Add a rainbow to this landscape",
 *     originalImages: [{
 *       url: "https://example.com/original.jpg",
 *       mimeType: "image/jpeg"
 *     }]
 *   });
 */
import { storagePut } from "server/storage";

export type GenerateImageOptions = {
  prompt: string;
  originalImages?: Array<{
    url?: string;
    b64Json?: string;
    mimeType?: string;
  }>;
  size?: "1024x1024" | "1792x1024" | "1024x1792";
  quality?: "standard" | "hd";
  style?: "vivid" | "natural";
};

export type GenerateImageResponse = {
  url?: string;
};

// Determine which API to use based on environment variables
const getImageApiConfig = () => {
  // Priority 1: OpenAI API (for self-hosted deployments)
  if (process.env.OPENAI_API_KEY) {
    return {
      provider: "openai" as const,
      apiKey: process.env.OPENAI_API_KEY,
    };
  }
  
  // Priority 2: Manus Forge API (for Manus-hosted deployments)
  const forgeApiUrl = process.env.BUILT_IN_FORGE_API_URL || process.env.FORGE_API_URL;
  const forgeApiKey = process.env.BUILT_IN_FORGE_API_KEY || process.env.FORGE_API_KEY;
  
  if (forgeApiKey && forgeApiUrl) {
    return {
      provider: "forge" as const,
      apiKey: forgeApiKey,
      apiUrl: forgeApiUrl,
    };
  }
  
  throw new Error("No API key configured. Set OPENAI_API_KEY for self-hosted or use Manus hosting.");
};

// Generate image using OpenAI DALL-E 3
async function generateWithOpenAI(
  options: GenerateImageOptions,
  apiKey: string
): Promise<GenerateImageResponse> {
  const response = await fetch("https://api.openai.com/v1/images/generations", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "dall-e-3",
      prompt: options.prompt,
      n: 1,
      size: options.size || "1024x1024",
      quality: options.quality || "standard",
      style: options.style || "vivid",
      response_format: "b64_json",
    }),
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(
      `OpenAI image generation failed (${response.status} ${response.statusText})${detail ? `: ${detail}` : ""}`
    );
  }

  const result = (await response.json()) as {
    data: Array<{
      b64_json: string;
      revised_prompt?: string;
    }>;
  };

  if (!result.data || result.data.length === 0) {
    throw new Error("No image generated");
  }

  const base64Data = result.data[0].b64_json;
  const buffer = Buffer.from(base64Data, "base64");

  // Save to S3
  const { url } = await storagePut(
    `generated/${Date.now()}.png`,
    buffer,
    "image/png"
  );

  return { url };
}

// Generate image using Manus Forge
async function generateWithForge(
  options: GenerateImageOptions,
  apiKey: string,
  apiUrl: string
): Promise<GenerateImageResponse> {
  const baseUrl = apiUrl.endsWith("/") ? apiUrl : `${apiUrl}/`;
  const fullUrl = new URL(
    "images.v1.ImageService/GenerateImage",
    baseUrl
  ).toString();

  const response = await fetch(fullUrl, {
    method: "POST",
    headers: {
      "Accept": "application/json",
      "Content-Type": "application/json",
      "connect-protocol-version": "1",
      "Authorization": `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      prompt: options.prompt,
      original_images: options.originalImages || [],
    }),
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(
      `Forge image generation failed (${response.status} ${response.statusText})${detail ? `: ${detail}` : ""}`
    );
  }

  const result = (await response.json()) as {
    image: {
      b64Json: string;
      mimeType: string;
    };
  };

  const base64Data = result.image.b64Json;
  const buffer = Buffer.from(base64Data, "base64");

  // Save to S3
  const { url } = await storagePut(
    `generated/${Date.now()}.png`,
    buffer,
    result.image.mimeType
  );

  return { url };
}

export async function generateImage(
  options: GenerateImageOptions
): Promise<GenerateImageResponse> {
  const config = getImageApiConfig();

  if (config.provider === "openai") {
    return generateWithOpenAI(options, config.apiKey);
  } else {
    return generateWithForge(options, config.apiKey, config.apiUrl!);
  }
}

// Export config checker for debugging
export function getImageGenConfig() {
  try {
    const config = getImageApiConfig();
    return {
      provider: config.provider === "openai" ? "OpenAI DALL-E 3" : "Manus Forge",
      configured: true,
    };
  } catch {
    return {
      provider: "None",
      configured: false,
    };
  }
}

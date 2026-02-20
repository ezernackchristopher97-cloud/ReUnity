CREATE TABLE `conversations` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`title` varchar(255),
	`currentState` varchar(32) DEFAULT 'stable',
	`currentRegime` varchar(32) DEFAULT 'normal',
	`isActive` boolean DEFAULT true,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `conversations_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `messages` (
	`id` int AUTO_INCREMENT NOT NULL,
	`conversationId` int NOT NULL,
	`role` enum('user','assistant') NOT NULL,
	`content` text NOT NULL,
	`entropyScore` varchar(10),
	`detectedState` varchar(32),
	`detectedPatterns` json,
	`groundingTechnique` varchar(64),
	`detectedLocation` varchar(64),
	`isCrisis` boolean DEFAULT false,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `messages_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `sessionAnalytics` (
	`id` int AUTO_INCREMENT NOT NULL,
	`conversationId` int NOT NULL,
	`userId` int NOT NULL,
	`messageCount` int DEFAULT 0,
	`crisisCount` int DEFAULT 0,
	`patternCount` int DEFAULT 0,
	`groundingCount` int DEFAULT 0,
	`avgEntropyScore` varchar(10),
	`durationSeconds` int,
	`finalState` varchar(32),
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `sessionAnalytics_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `userMemory` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`memoryType` varchar(32) NOT NULL,
	`memoryKey` varchar(128) NOT NULL,
	`memoryValue` text NOT NULL,
	`confidence` varchar(10) DEFAULT '1.0',
	`accessCount` int DEFAULT 0,
	`lastAccessed` timestamp DEFAULT (now()),
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `userMemory_id` PRIMARY KEY(`id`)
);

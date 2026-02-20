CREATE TABLE `journalEntries` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`title` varchar(255),
	`content` text NOT NULL,
	`moodTags` json,
	`customTags` json,
	`entropyScore` varchar(10),
	`entropyState` varchar(32),
	`detectedConditions` json,
	`detectedStates` json,
	`triggersIdentified` json,
	`copingUsed` json,
	`isPrivate` boolean DEFAULT true,
	`isEncrypted` boolean DEFAULT false,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `journalEntries_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `journalInsights` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`insightType` enum('pattern','progress','warning','suggestion') NOT NULL,
	`title` varchar(255) NOT NULL,
	`description` text NOT NULL,
	`confidence` varchar(10),
	`relatedEntries` json,
	`isDismissed` boolean DEFAULT false,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `journalInsights_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `moderationActions` (
	`id` int AUTO_INCREMENT NOT NULL,
	`targetUserId` int NOT NULL,
	`reporterId` int,
	`action` enum('warning','temporary_ban','permanent_ban','review_required') NOT NULL,
	`reason` text NOT NULL,
	`resolvedAt` timestamp,
	`resolvedBy` int,
	`resolution` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `moderationActions_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `peerConnections` (
	`id` int AUTO_INCREMENT NOT NULL,
	`requesterId` int NOT NULL,
	`responderId` int NOT NULL,
	`status` enum('pending','accepted','declined','blocked','ended') NOT NULL DEFAULT 'pending',
	`matchScore` int,
	`sharedExperiences` json,
	`sessionCount` int DEFAULT 0,
	`lastSessionAt` timestamp,
	`totalMinutes` int DEFAULT 0,
	`flaggedForReview` boolean DEFAULT false,
	`flagReason` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `peerConnections_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `peerMessages` (
	`id` int AUTO_INCREMENT NOT NULL,
	`connectionId` int NOT NULL,
	`senderId` int NOT NULL,
	`content` text NOT NULL,
	`entropyLevel` varchar(10),
	`crisisDetected` boolean DEFAULT false,
	`flagged` boolean DEFAULT false,
	`flagReason` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `peerMessages_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `peerProfiles` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`displayName` varchar(64) NOT NULL,
	`experiences` json,
	`preferences` json,
	`safetySettings` json,
	`isActive` boolean DEFAULT true,
	`lastActive` timestamp DEFAULT (now()),
	`isBanned` boolean DEFAULT false,
	`banReason` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `peerProfiles_id` PRIMARY KEY(`id`),
	CONSTRAINT `peerProfiles_userId_unique` UNIQUE(`userId`)
);
--> statement-breakpoint
CREATE TABLE `safetyPlans` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`encryptedData` text NOT NULL,
	`completedSteps` json,
	`isComplete` boolean DEFAULT false,
	`lastStepId` varchar(64),
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `safetyPlans_id` PRIMARY KEY(`id`)
);
